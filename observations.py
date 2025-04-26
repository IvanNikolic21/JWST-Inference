import numpy as np

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

            self.theta_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_theta.dat"
                ).readlines()
            ]
            self.wtheta_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_wtheta.dat"
                ).readlines()
            ]
            self.wsig_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_wsig.dat"
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

    def get_obs_z7_m90(self):
        thethats90_pos_z7 = [
            float(i) for i,j in zip(self.theta_z7[3],self.wtheta_z7[3]) if float(j)>0
        ]
        wthethats90_pos_z7 = [
            float(i) for i,j in zip(self.wtheta_z7[2],self.wtheta_z7[2]) if float(j)>0
        ]
        wsig90_pos_z7 = [
            float(i) for i,j in zip(self.wsig_z7[2],self.wtheta_z7[2]) if float(j)>0
        ]
        return thethats90_pos_z7, wthethats90_pos_z7, wsig90_pos_z7

    def get_obs_uvlf_z11_McLeod23(self):
        McL_Muvs = np.array(
            [-22.57,-21.80,-20.80,-20.05,-19.55,-18.85,-18.23]
        )
        McL_uvlf = np.array(
            [0.012,0.128,1.251,3.951,9.713,23.490,63.080]
        )*1e-5
        Mcl_sig = np.array(
            [0.010,0.128,0.424,1.319, 4.170,9.190,28.650]
        )*1e-5

        return McL_Muvs, McL_uvlf, Mcl_sig

    def get_obs_uvlf_z9_Donnan24(self):
        Don_Muvs = np.array(
            [-20.75,-20.25,-19.75,-19.25,-18.55,-18.05,-17.55]
        )
        Don_uvlf = np.array(
            [12,32,144,235,486,1110,1776]
        )*1e-6
        Don_sig_p = np.array(
            [8,13,30,60,157,310,578]
        )*1e-6
        Don_sig_m = np.array(
            [5,10,28,49,139,310,510]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)

    def get_obs_uvlf_z10_Donnan24(self):
        Don_Muvs = np.array(
            [-20.75,-20.25,-19.75,-19.25,-18.55,-18.05,-17.55]
        )
        Don_uvlf = np.array(
            [4,27,92,177,321,686,1278]
        )*1e-6
        Don_sig_p = np.array(
            [10,13,25,53,127,245,486]
        )*1e-6
        Don_sig_m = np.array(
            [4,10,20,45,111,223,432]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)

    def get_obs_uvlf_z11_Donnan24(self):
        Don_Muvs = np.array(
            [-21.25,-20.75,-20.25,-19.75,-19.25,-18.75,-18.25]
        )
        Don_uvlf = np.array(
            [7,14,38,100,144,234,641]
        )*1e-6
        Don_sig_p = np.array(
            [9,11,16,37,81,118,361]
        )*1e-6
        Don_sig_m = np.array(
            [5,7,13,30,63,96,281]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)

    def get_obs_uvlf_z12_5_Donnan24(self):
        Don_Muvs = np.array(
            [-21.25,-20.75,-20.25,-19.75,-19.25,-18.75,-18.25]
        )
        Don_uvlf = np.array(
            [3,4,16,34,43,80,217]
        )*1e-6
        Don_sig_p = np.array(
            [4,5,9,23,35,51,153]
        )*1e-6
        Don_sig_m = np.array(
            [2,3,6,15,22,36,104]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)

    def get_obs_uvlf_z9_Harikane24(self):
        Har_Muvs = np.array(
            [-22.0, -21.0, -20.0, -19.0]
        )
        Har_uvlf = np.array(
            [6.6 * 1e-6,5.1 * 1e-6,2.9*1e-5,3.5*1e-5]
        )
        Har_sig_p = np.array(
            [7.1 * 1e-6,-1* 7.0*1e-6, -1*3.2 * 1e-5, -1*3.7 * 1e-5]
        )  #trick for lower limit
        Har_sig_m = np.array(
            [4.7 * 1e-6, -1* 3.8 * 1e-6, -1* 2.2 * 1e-5, -1* 2.4 * 1e-5]
        )
        return Har_Muvs, Har_uvlf, (Har_sig_p, Har_sig_m)

    def get_obs_uvlf_z10_Harikane24(self):
        Har_Muvs = np.array(
            [-21.6, -20.6, -19.6,-18.6,-17.6]
        )
        Har_uvlf = np.array(
            [1.0 * 1e-6,8.7 * 1e-6,2.6*1e-5,1.9*1e-4, 6.3*1e-4]
        )
        Har_sig_p = np.array(
            [2.3 * 1e-6, -1 * 20.5*1e-6, -1 * 2.8 * 1e-5, 4.7 * 1e-4, 15.8 * 1e-4]
        )  #trick for lower limit
        Har_sig_m = np.array(
            [0.9 * 1e-6, -1 * 8.4 * 1e-6, -1 * 1.8 * 1e-5, 1.9 * 1e-4, 6.3 * 1e-4]
        )
        return Har_Muvs, Har_uvlf, (Har_sig_p, Har_sig_m)

    def get_obs_uvlf_z12_Harikane24(self):
        Har_Muvs = np.array(
            [-20.5, -20.1, -18.7]
        )
        Har_uvlf = np.array(
            [8.6 * 1e-6,8.8 * 1e-6,6.6*1e-5]
        )
        Har_sig_p = np.array(
            [19.9 * 1e-6,-1* 9.1*1e-6, -1*6.0 * 1e-5]
        )  #trick for lower limit
        Har_sig_m = np.array(
            [7.4 * 1e-6, -1* 5.5 * 1e-6, -1* 4.6 * 1e-5]
        )
        return Har_Muvs, Har_uvlf, (Har_sig_p, Har_sig_m)

    def get_obs_uvlf_z14_Harikane24(self):
        Har_Muvs = np.array(
            [-20.8, -19.0]
        )
        Har_uvlf = np.array(
            [3.7 * 1e-5,3.7*1e-5]
        )
        Har_sig_p = np.array(
            [8.7*1e-5, -1*8.7 * 1e-5]
        )  #trick for lower limit
        Har_sig_m = np.array(
            [3.6 * 1e-5, -1* 3.6 * 1e-5]
        )
        return Har_Muvs, Har_uvlf, (Har_sig_p, Har_sig_m)
