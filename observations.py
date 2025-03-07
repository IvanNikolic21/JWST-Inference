dir_dat = "/home/inikolic/projects/UVLF_FMs/data/Paquereau_2025_clustering/GalClustering_COSMOS-Web_Paquereau2025/clustering_measurements/"


class Observations():
    def __init__(self, ang, uvlf):
        if ang:
            self.cons_theta = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_theta.dat"
                ).readlines()
            ]
            self.cons_wtheta = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wtheta.dat"
                ).readlines()
            ]
            self.cons_wsig = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wsig.dat"
                ).readlines()
            ]
        if uvlf:
            pass

    def get_obs_z9_m87(self):
        thethats87_pos = [
            float(i) for i,j in zip(self.cons_theta[1],self.cons_wtheta[1]) if float(j)>0
        ]
        wthethats87_pos = [
            float(i) for i,j in zip(self.cons_wtheta[1],self.cons_wtheta[1]) if float(j)>0
        ]
        wsig87_pos = [
            float(i) for i,j in zip(self.cons_wsig[1],self.cons_wtheta[1]) if float(j)>0
        ]
        return thethats87_pos, wthethats87_pos, wsig87_pos

    def get_obs_z9_m90(self):
        thethats90_pos = [
            float(i) for i,j in zip(self.cons_theta[2],self.cons_wtheta[2]) if float(j)>0
        ]
        wthethats90_pos = [
            float(i) for i,j in zip(self.cons_wtheta[2],self.cons_wtheta[2]) if float(j)>0
        ]
        wsig90_pos = [
            float(i) for i,j in zip(self.cons_wsig[2],self.cons_wtheta[2]) if float(j)>0
        ]
        return thethats90_pos, wthethats90_pos, wsig90_pos



