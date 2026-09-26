"""Export FIRE-2 High Redshift suite galaxies to prior_pipeline's catalog format.

The public Rockstar catalogs give halo mass and stellar mass but no SFR, so
the SFR is computed from the formation times of each galaxy's star
particles, which are read from the public snapshots over HTTPS. Only the two
star-particle datasets needed (StellarFormationTime, Masses) are fetched,
via HTTP range requests, instead of the 2-9 GB snapshots.

Defaults: the 22 z5* simulations (the only ones with Rockstar catalogs) at
snapshots 67, 41, 26 (z = 5, 7, 9); halos with M_vir > 1e9 Msun and
low-resolution mass fraction < 1%; SFR averaged over the last 100 Myr.

NOTE: star-particle masses are current masses, not masses at formation, so
SFRs are low by the fraction of mass returned by stellar evolution within
the averaging window. Rockstar subhalos are kept alongside hosts.

Usage:
  python -m prior_pipeline.export_fire2 --out prior_pipeline/data/fire_catalog.json
"""
import argparse
import io
import json
import os
import re
import ssl
import urllib.request
from collections import OrderedDict

import numpy as np
from astropy.cosmology import FlatLambdaCDM

BASE_URL = ("https://users.flatironinstitute.org/~mgrudic/"
            "fire2_public_release/high_redshift")
Z5_SIMS = ["z5m09a", "z5m09b",
           "z5m10a", "z5m10b", "z5m10c", "z5m10d", "z5m10e", "z5m10f",
           "z5m11a", "z5m11b", "z5m11c", "z5m11d", "z5m11e", "z5m11f",
           "z5m11g", "z5m11h", "z5m11i",
           "z5m12a", "z5m12b", "z5m12c", "z5m12d", "z5m12e"]
MASS_UNIT = 1e10  # snapshot masses are in 1e10 Msun/h


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, with a block
    cache. Lets h5py open a remote HDF5 file and fetch only what it reads."""

    def __init__(self, url, block_size=4 * 2 ** 20, max_blocks=64):
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._cache = OrderedDict()
        self._pos = 0
        self.bytes_fetched = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=_ssl_context()) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}
        self._pos = base[whence] + offset
        return self._pos

    def readinto(self, buf):
        n = min(len(buf), max(self.size - self._pos, 0))
        out = memoryview(buf)
        done = 0
        while done < n:
            block, offset = divmod(self._pos + done, self.block_size)
            data = self._block(block)
            chunk = data[offset:offset + n - done]
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
        self._pos += n
        return n

    def _block(self, i):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        start = i * self.block_size
        end = min(start + self.block_size, self.size) - 1
        req = urllib.request.Request(self.url,
                                     headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, context=_ssl_context()) as r:
            data = r.read()
        self.bytes_fetched += len(data)
        self._cache[i] = data
        if len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
        return data


def _ssl_context():
    if os.environ.get("FIRE_INSECURE_SSL"):
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def _list_dir(url):
    with urllib.request.urlopen(url, context=_ssl_context()) as r:
        html = r.read().decode()
    return set(re.findall(r'href="([^"?/][^"]*)"', html))


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, context=_ssl_context()) as r, \
                open(path, "wb") as f:
            f.write(r.read())
    return path


def snapshot_urls(sim, snap):
    """URLs of the snapshot's file(s): a single file or a snapdir."""
    out = f"{BASE_URL}/{sim}/output/"
    entries = _list_dir(out)
    if f"snapshot_{snap:03d}.hdf5" in entries:
        return [out + f"snapshot_{snap:03d}.hdf5"]
    snapdir = out + f"snapdir_{snap:03d}/"
    files = sorted((f for f in _list_dir(snapdir) if f.endswith(".hdf5")),
                   key=lambda f: int(f.split(".")[-2]))
    return [snapdir + f for f in files]


def read_star_particles(urls):
    """Formation scale factor and mass [1e10 Msun/h] of all star particles,
    concatenated in file order (the order star.indices refers to)."""
    import h5py

    a_form, mass, fetched = [], [], 0
    for url in urls:
        rf = HTTPRangeFile(url)
        with h5py.File(rf, "r") as f:
            if "PartType4" in f:
                a_form.append(f["PartType4/StellarFormationTime"][:])
                mass.append(f["PartType4/Masses"][:])
        fetched += rf.bytes_fetched
    return np.concatenate(a_form), np.concatenate(mass), fetched


def export_snapshot(sim, snap, cache_dir, sfr_timescale_myr, min_mh,
                    max_lowres):
    import h5py

    cat = f"{BASE_URL}/{sim}/halo/rockstar_dm/catalog_hdf5/"
    halo_path = _download(cat + f"halo_{snap:03d}.hdf5",
                          os.path.join(cache_dir, sim, f"halo_{snap:03d}.hdf5"))
    star_path = _download(cat + f"star_{snap:03d}.hdf5",
                          os.path.join(cache_dir, sim, f"star_{snap:03d}.hdf5"))
    with h5py.File(halo_path, "r") as h, h5py.File(star_path, "r") as s:
        z = float(h["snapshot:redshift"][()])
        hubble = float(h["cosmology:hubble"][()])
        cosmo = FlatLambdaCDM(H0=100 * hubble,
                              Om0=float(h["cosmology:omega_matter"][()]))
        mh = h["mass.vir"][:].astype(np.float64)
        lowres = h["mass.lowres"][:] / h["mass"][:]
        ms = s["star.mass"][:].astype(np.float64)
        keep = np.flatnonzero((mh > min_mh) & (lowres < max_lowres) & (ms > 0))
        indices = [s["star.indices"][i] for i in keep]

    a_form, mass, fetched = read_star_particles(snapshot_urls(sim, snap))
    t_now = cosmo.age(z).to("Myr").value
    a_grid = np.linspace(a_form.min(), 1.0 / (1.0 + z), 2000)
    t_form = np.interp(a_form, a_grid,
                       cosmo.age(1.0 / a_grid - 1.0).to("Myr").value)
    young = (t_now - t_form) < sfr_timescale_myr
    msun = mass * MASS_UNIT / hubble
    sfr = np.array([msun[idx][young[idx]].sum() for idx in indices])
    sfr /= sfr_timescale_myr * 1e6
    print(f"  {sim} snap {snap} (z={z:.2f}): {len(keep)} galaxies, "
          f"{fetched / 1e6:.1f} MB streamed")
    return z, mh[keep], ms[keep], sfr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", nargs="+", default=Z5_SIMS)
    ap.add_argument("--snapshots", nargs="+", type=int, default=[67, 41, 26])
    ap.add_argument("--sfr-timescale", type=float, default=100.0, help="Myr")
    ap.add_argument("--min-mh", type=float, default=1e9)
    ap.add_argument("--max-lowres", type=float, default=0.01)
    ap.add_argument("--cache-dir", default="fire2_catalog_cache")
    args = ap.parse_args()

    per_snap = {}
    for snap in args.snapshots:
        print(f"snapshot {snap}")
        for sim in args.sims:
            z, mh, ms, sfr = export_snapshot(sim, snap, args.cache_dir,
                                             args.sfr_timescale, args.min_mh,
                                             args.max_lowres)
            d = per_snap.setdefault(snap, {"z": [], "mh": [], "ms": [], "sfr": []})
            d["z"].append(z)
            d["mh"].append(mh)
            d["ms"].append(ms)
            d["sfr"].append(sfr)

    snapshots = []
    for snap, d in per_snap.items():
        snapshots.append({
            "z": round(float(np.mean(d["z"])), 3),
            "log10_mh": np.log10(np.concatenate(d["mh"])).tolist(),
            "log10_ms": np.log10(np.concatenate(d["ms"])).tolist(),
            "sfr": np.concatenate(d["sfr"]).tolist(),
        })
        print(f"z={snapshots[-1]['z']}: {len(snapshots[-1]['sfr'])} galaxies")
    with open(args.out, "w") as f:
        json.dump({"description": (
            "FIRE-2 High Redshift suite (Ma+2018-2020), Rockstar catalogs + "
            f"SFR over {args.sfr_timescale:g} Myr from star particles; "
            f"sims {args.sims}; snapshots {args.snapshots}; "
            f"M_vir > {args.min_mh:g}, lowres < {args.max_lowres:g}"),
            "snapshots": snapshots}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
