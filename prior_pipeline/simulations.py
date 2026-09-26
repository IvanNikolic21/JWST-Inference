"""Load simulated galaxy catalogs into a common format.

Each loader returns a Simulation: a list of Snapshots holding (Mh, M*, SFR)
per galaxy at a given redshift. Mh is None for simulations without halo
masses, which then only constrain the SFMS.

The selections reproduce basic_prior_setup.ipynb. Where the notebook made a
choice that may be worth revisiting, it is kept for now and marked NOTE.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Snapshot:
    z: float
    ms: np.ndarray            # Msun
    sfr: np.ndarray           # Msun/yr
    mh: np.ndarray = None     # Msun, or None


@dataclass
class Simulation:
    name: str
    snapshots: list = field(default_factory=list)

    @property
    def has_halos(self):
        return all(s.mh is not None for s in self.snapshots)

    def summary(self):
        return ", ".join(f"z={s.z:g}: {len(s.ms)} gal" for s in self.snapshots)


def load_catalog_file(name, path):
    """Generic JSON catalog: {"snapshots": [{"z", "log10_ms", "sfr",
    optional "log10_mh"}]}. Used for SERRA and for any simulation exported
    to this format (e.g. FIRE)."""
    with open(_resolve(path)) as f:
        cat = json.load(f)
    snaps = []
    for s in cat["snapshots"]:
        mh = 10 ** np.asarray(s["log10_mh"]) if "log10_mh" in s else None
        snaps.append(Snapshot(z=s["z"], ms=10 ** np.asarray(s["log10_ms"]),
                              sfr=np.asarray(s["sfr"], dtype=np.float64),
                              mh=mh))
    return Simulation(name, snaps)


def load_firstlight(name, database, ids="firstlight_ids.json",
                    target_z=(5.0, 10.0, 15.0)):
    """First Light database (Ceverino+2018): for each target redshift, take
    the snapshot with the closest scale factor and keep unique
    (Mvir, SFR, Ms) triples.

    NOTE: the SFMS is evaluated at the target z, not at the snapshot's
    actual 1/a - 1 (e.g. 15.13 for z=15), as in the notebook.
    """
    with open(_resolve(database)) as f:
        db = json.load(f)
    with open(_resolve(ids)) as f:
        fl_ids = json.load(f)
    possible_a = np.array([0.059, 0.060, 0.062, 0.064, 0.067, 0.069, 0.071,
                           0.074, 0.077, 0.080, 0.083, 0.087, 0.090, 0.095,
                           0.1, 0.105, 0.111, 0.118, 0.125, 0.133, 0.142,
                           0.154, 0.160])
    z_poss = 1.0 / possible_a - 1
    snaps = []
    for zt in target_z:
        a_str = "{:.3f}".format(possible_a[np.argmin(np.abs(z_poss - zt))])
        triples = set()
        for gid in fl_ids:
            key = f"{gid}_{a_str}_"
            if key + "Mvir" in db and key + "Ms" in db:
                triples.add((db[key + "Mvir"], db[key + "SFR"], db[key + "Ms"]))
        mh, sfr, ms = (np.array(x, dtype=np.float64)
                       for x in zip(*sorted(triples)))
        snaps.append(Snapshot(z=zt, ms=ms, sfr=sfr, mh=mh))
    return Simulation(name, snaps)


def load_astrid(name, snapshots, n_per_bin=10, n_bins=10, log10_mh_min=8.3,
                seed=0):
    """ASTRID FOF groups (Bird+2022): in n_bins log-spaced halo-mass bins,
    draw n_per_bin halos (with replacement) per bin.

    Each entry of `snapshots` is {"path", "z", optional "n_rows",
    optional "log10_mh_max"}. Without log10_mh_max the upper bin edge is the
    most massive halo, as in the notebook.

    NOTE: like the notebook, halo mass is the DM-only MassByType[:, 1], and
    np.digitize index 0 (halos below the first edge) is used as a bin while
    halos above the last edge are dropped. The notebook assigned z=5 to the
    PIG_035_z5-5 snapshot; the z in the config is used as given.
    """
    import bigfile

    rng = np.random.default_rng(seed)
    snaps = []
    for spec in snapshots:
        f = bigfile.File(_resolve(spec["path"]))
        n = spec.get("n_rows", f["MassByType"].size)
        mbt = np.array(f["MassByType"][:n])
        hm = 1e10 * mbt[:, 1]
        sm = 1e10 * mbt[:, 4]
        sfr = np.array(f["StarFormationRate"][:n])
        f.close()

        log_max = spec.get("log10_mh_max", np.log10(hm.max()))
        edges = np.logspace(log10_mh_min, log_max, n_bins)
        idx = np.digitize(hm, edges)
        pick = []
        for i in range(n_bins):
            members = np.flatnonzero(idx == i)
            if len(members):
                pick.append(rng.choice(members, n_per_bin))
        pick = np.concatenate(pick)
        snaps.append(Snapshot(z=spec["z"], ms=sm[pick], sfr=sfr[pick],
                              mh=hm[pick]))
    return Simulation(name, snaps)


def load_flares(name, path, snapshots, region="00", ms_unit=1e10):
    """FLARES (Lovell+2021) public galaxy catalog; no halo masses, so it
    only constrains the SFMS. `snapshots` maps z -> snapshot tag, e.g.
    {"5": "010_z005p000", "10": "005_z010p000"}.

    NOTE: Mstar_30 is stored in units of 1e10 Msun (ms_unit). The notebook
    that fit FLARES is not in basic_prior_setup.ipynb; check this matches.
    """
    import h5py

    snaps = []
    with h5py.File(_resolve(path), "r") as f:
        for z, tag in snapshots.items():
            gal = f[region][tag]["Galaxy"]
            snaps.append(Snapshot(z=float(z),
                                  ms=np.array(gal["Mstar_30"]) * ms_unit,
                                  sfr=np.array(gal["SFR_inst_30"])))
    return Simulation(name, snaps)


LOADERS = {
    "catalog_file": load_catalog_file,
    "firstlight": load_firstlight,
    "astrid": load_astrid,
    "flares": load_flares,
}


def load_simulation(name, spec):
    """Load a simulation from its config entry. Keys starting with '_' are
    comments; fit_bounds, legacy_posterior and halo_masses are used by
    build_priors.py, not by the loaders."""
    skip = {"loader", "fit_bounds", "legacy_posterior", "halo_masses"}
    kwargs = {k: v for k, v in spec.items()
              if k not in skip and not k.startswith("_")}
    return LOADERS[spec["loader"]](name, **kwargs)


def _resolve(path):
    """Relative paths are looked up in prior_pipeline/data."""
    return path if os.path.isabs(path) else os.path.join(DATA_DIR, path)
