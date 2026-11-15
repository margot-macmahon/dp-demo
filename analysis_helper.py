import glob
import math

# from tqdm import tqdm
import pickle
from collections.abc import Sequence
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter

# TODO replace arrays assumed to match order with parameters etc with dicts


# --- Resolution model (absolute sigma in GeV) ---
def sigma(E_true: float) -> float:
    """
    Absolute energy resolution in GeV:
      sigma(E) = pars[0]*E + pars[1]*sqrt(E) + pars[2]
    Notes:
      - E_true must be >= 0 for sqrt(E_true).
      - pars = (a, b, c) with units chosen so that sigma is in GeV.
    """
    if E_true < 0:
        raise ValueError("E_true must be non-negative for sqrt(E_true).")
    a, b, c = [0.07453287, 0.10387965, 0.00224152]
    return a * E_true + b * math.sqrt(max(E_true, 0.0)) + c


def glb_upper_bin_boundary(i: int, data) -> float:
    return data["Ereco_edges"][i + 1]


def glb_lower_bin_boundary(i: int, data) -> float:
    return data["Ereco_edges"][i]


def glb_sbin_center(j: int, data) -> float:
    return data["Etrue_centers"][j]


def Sigma(j: int, data) -> float:
    # Sigma evaluated at the true-energy bin center
    E = glb_sbin_center(j, data)
    return sigma(E)


def SmearMatrixElementA(i: int, j: int, data) -> float:
    """
    Probability that a Gaussian-smeared measurement (mean = E_true bin center j,
    width = Sigma(j)) falls inside reco bin i.
    """
    center = glb_sbin_center(j, data)
    s = Sigma(j, data)
    if s <= 0:
        # Degenerate case: no smearing -> probability is 1 if center in bin, else 0
        lo = glb_lower_bin_boundary(i, data)
        hi = glb_upper_bin_boundary(i, data)
        return 1.0 if (lo <= center < hi) else 0.0

    inv = 1.0 / (math.sqrt(2.0) * s)

    u = glb_upper_bin_boundary(i, data)
    l = glb_lower_bin_boundary(i, data)

    p1 = 0.5 * (1.0 + math.erf((u - center) * inv))
    p2 = 0.5 * (1.0 + math.erf((l - center) * inv))
    return p1 - p2


def get_bin_edges(bin_centres, bin_width):
    bin_edges = [bin_centres[0] - bin_width]
    bin_edges.extend(bin_centres + bin_width)
    return bin_edges


def build_smear_matrix(Etrue_centers: Sequence[float]):
    """
    Construct S[i][j] = P(E_reco in bin i | E_true at center of bin j), using a Gaussian with
    sigma given by sigma_pars.

    Args:
      Ereco_edges: length N_reco+1 array of reconstructed-energy bin edges (GeV).
      Etrue_centers: length N_true array of true-energy bin centers (GeV).
      sigma_pars: (a, b, c) parameters for sigma(E) in GeV:
                  sigma(E) = a*E + b*sqrt(E) + c

    Returns:
      S: list-of-lists of shape (N_reco, N_true)
    """
    Ereco_edges = get_bin_edges(Etrue_centers, 0.125)
    data = {"Ereco_edges": list(Ereco_edges), "Etrue_centers": list(Etrue_centers)}
    n_reco = len(Ereco_edges) - 1
    n_true = len(Etrue_centers)
    S = [[0.0] * n_true for _ in range(n_reco)]
    for i in range(n_reco):
        for j in range(n_true):
            S[i][j] = SmearMatrixElementA(i, j, data)
    return S


# load xsec, digitalized from 1305.7513, sigma/E (10^(-38)cm^2/GeV)
def xsecs_base(
    Enu=np.arange(0.125, 10.625, 0.25),
    Rqe=1,
    Rres=1,
    Rdis=1,
    Rmec=1,
    gammaqel=0,
    gammares=0,
    gammadis=0,
    gammamec=0,
    id="",
    param=True,
):
    dis_scale, res_scale, qel_scale, mec_scale = (
        np.ones_like(Enu),
        np.ones_like(Enu),
        np.ones_like(Enu),
        np.ones_like(Enu),
    )  # TODO redefine R_i to be 1 and i_scale to be nominally 1, adjusted for nue xsecs.
    base_path = "genie_xsecs/"
    xsec_files = {
        "numu": {
            "qe": "numu_qel_cc_",
            "dis": "numu_dis_cc_",
            "res": "numu_res_cc_",
            "mec": "numu_mec_cc_",
        },
        "a_numu": {
            "qe": "a_numu_qel_cc_",
            "dis": "a_numu_dis_cc_",
            "res": "a_numu_res_cc_",
            "mec": "a_numu_mec_cc_",
        },
        "nue": {
            "qe": "nue_qel_cc_",
            "dis": "nue_dis_cc_",
            "res": "nue_res_cc_",
            "mec": "nue_mec_cc_",
        },
        "a_nue": {
            "qe": "a_nue_qel_cc_",
            "dis": "a_nue_dis_cc_",
            "res": "a_nue_res_cc_",
            "mec": "a_nue_mec_cc_",
        },
    }

    if "anti" in id:
        xsec_type = "a_numu"
        if id == "anti_electron":
            if param:
                dis_scale = np.loadtxt(
                    base_path + xsec_files["a_nue"]["dis"] + "rat.txt"
                )
                dis_scale = np.interp(Enu, dis_scale[:, 0], dis_scale[:, 1])
                res_scale = np.loadtxt(
                    base_path + xsec_files["a_nue"]["res"] + "rat.txt"
                )
                res_scale = np.interp(Enu, res_scale[:, 0], res_scale[:, 1])
                qel_scale = np.loadtxt(
                    base_path + xsec_files["a_nue"]["qe"] + "rat.txt"
                )
                qel_scale = np.interp(Enu, qel_scale[:, 0], qel_scale[:, 1])
                mec_scale = np.loadtxt(
                    base_path + xsec_files["a_nue"]["mec"] + "rat.txt"
                )
                mec_scale = np.interp(
                    Enu, mec_scale[:, 0], mec_scale[:, 1]
                )  # 0.9955, 1.0222, 1.0005, 1.0106
            else:
                xsec_type = "a_nue"
    elif id == "electron":
        xsec_type = "numu"
        if param:
            dis_scale = np.loadtxt(base_path + xsec_files["nue"]["dis"] + "rat.txt")
            dis_scale = np.interp(Enu, dis_scale[:, 0], dis_scale[:, 1])
            res_scale = np.loadtxt(base_path + xsec_files["nue"]["res"] + "rat.txt")
            res_scale = np.interp(Enu, res_scale[:, 0], res_scale[:, 1])
            qel_scale = np.loadtxt(base_path + xsec_files["nue"]["qe"] + "rat.txt")
            qel_scale = np.interp(Enu, qel_scale[:, 0], qel_scale[:, 1])
            mec_scale = np.loadtxt(base_path + xsec_files["nue"]["mec"] + "rat.txt")
            mec_scale = np.interp(
                Enu, mec_scale[:, 0], mec_scale[:, 1]
            )  # 0.9922, 1.0318, 1.0247, 1.0448
            # print(mec_scale)
        else:
            xsec_type = "nue"
    else:
        xsec_type = "numu"

    nuxsecqe = np.loadtxt(base_path + xsec_files[xsec_type]["qe"] + "tot.txt")
    nuxsecdis = np.loadtxt(base_path + xsec_files[xsec_type]["dis"] + "tot.txt")
    nuxsecres = np.loadtxt(base_path + xsec_files[xsec_type]["res"] + "tot.txt")
    nuxsecmec = np.loadtxt(base_path + xsec_files[xsec_type]["mec"] + "tot.txt")

    E0 = {"qe": 2.5, "dis": 2.5, "res": 2.5, "mec": 2.5}
    xsec_parts = {
        "qe": np.interp(Enu, nuxsecqe[:, 0], nuxsecqe[:, 1])
        * (Rqe * qel_scale)
        * (Enu / E0["qe"]) ** gammaqel
        * Enu,
        "dis": np.interp(Enu, nuxsecdis[:, 0], nuxsecdis[:, 1])
        * (Rdis * dis_scale)
        * (Enu / E0["dis"]) ** gammadis
        * Enu,
        "res": np.interp(Enu, nuxsecres[:, 0], nuxsecres[:, 1])
        * (Rres * res_scale)
        * (Enu / E0["res"]) ** gammares
        * Enu,
        "mec": np.interp(Enu, nuxsecmec[:, 0], nuxsecmec[:, 1])
        * (Rmec * mec_scale)
        * (Enu / E0["mec"]) ** gammamec
        * Enu,
    }

    xsectotal = (
        xsec_parts["qe"] + xsec_parts["dis"] + xsec_parts["res"] + xsec_parts["mec"]
    ) * 1e-38

    if "nc" in id:
        xsec_nc = np.loadtxt("globes_xsecs/xsec_nc.dat")
        nc_x, nc_y = 10 ** xsec_nc[:, 0], xsec_nc[:, 1] * 10 ** xsec_nc[:, 0]
        if id == "anc":
            nc_y = xsec_nc[:, 4] * 10 ** xsec_nc[:, 0]
        xsectotal = np.interp(Enu, nc_x, nc_y) * 1e-38

    return xsectotal


def initialize_flux_dir(dir):
    global flux_dir
    flux_dir = dir


def initialize_scaling(
    POT,
):  # sets total POT of which each off axis position will get a share
    global scaling_val
    scaling_val = 1e30 * POT  # pot * nAr_nucleons


def initialize_xsecs(param_bool):
    global xsecs
    xsecs = partial(xsecs_base, param=param_bool)


# fit a function to fluxes
def makepolynomfitfluxes_base(
    smear=True, decayid=0, theta=0.0, is_mu=True, contrib="pi"
):
    proc_names = np.array(
        [
            "kl_to_nue",
            "kl_to_nuebar",
            "kl_to_numu",
            "kl_to_numubar",
            "kp_to_numu_2b",
            "kp_to_nue",
            "kp_to_numu_3b",
            "km_to_numubar_2b",
            "km_to_nuebar",
            "km_to_numubar_3b",
            "mup_to_numubar",
            "mum_to_numu",
            "pip_to_numu",
            "pim_to_numubar",
        ]
    )
    flux = []
    # for i in range(len(proc_names)):
    #    flux=np.loadtxts
    if smear:
        smearval = 0.8
        non_smearval = 0.8
    else:
        smearval = 0
        non_smearval = 0
    # if decayid == 11 or decayid == 12:
    if is_mu:
        flux = np.loadtxt(
            flux_dir
            + proc_names[decayid - 1]
            + "_offaxis"
            + str(theta)
            + "_"
            + contrib
            + ".txt"
        )
    else:
        flux = np.loadtxt(
            flux_dir + proc_names[decayid - 1] + "_offaxis" + str(theta) + ".txt"
        )

    if len(flux) > 0:
        fluxin = flux[:, 1]
    else:
        return 0

    if (np.round(sum(flux[:, 1]), -3)) <= 100000:
        fluxfiltered = gaussian_filter(fluxin, sigma=smearval)  # 1)
    else:
        fluxfiltered = gaussian_filter(fluxin, sigma=non_smearval)
    fluxe = []
    fluxx = []
    flux_unfilt = []
    for i in range(len(flux)):
        # if(flux[i,1] !=0):
        fluxe.append(flux[i, 0])
        fluxx.append((fluxfiltered[i]))
        flux_unfilt.append((flux[i, 1]))
    t, c, k = interpolate.splrep(fluxe, fluxx)
    spline = interpolate.BSpline(t, c, k)
    if decayid == 20 and theta > 10:
        plt.plot(fluxe, flux_unfilt, label=str(decayid) + " " + str(theta))
        plt.plot(fluxe, fluxx)
        plt.ylim(0, 200)
        plt.legend()
        plt.show()
    return spline


def initialize_fluxes(smear_bool):
    global makepolynomfitfluxes
    makepolynomfitfluxes = partial(makepolynomfitfluxes_base, smear=smear_bool)


def makeflux(decayid, theta, Enu, is_mu, contrib=""):
    flux1 = makepolynomfitfluxes(
        decayid=decayid, theta=theta, is_mu=is_mu, contrib=contrib
    )
    if flux1 == 0:
        return 0
    else:
        return np.array(flux1(Enu))


def get_cov_mat(m_file, flav):
    names = [
        "kl",
        "kp",
        "km",
        "pip",
        "pim",
        "rqe",
        "rres",
        "rdis",
        "rmec",
        "gammaqe",
        "gammares",
        "gammadis",
        "gammamec",
    ]
    with open(m_file, "rb") as f:
        m = pickle.load(f)
    mat = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            mat[i][j] = m.covariance[i][j]
    md = {}
    for i in range(len(names)):
        md[names[i]] = mat[i]
    if flav == "tot":
        cov_a = np.array(
            [
                md["kl"],
                md["kl"],
                md["kl"],
                md["kl"],
                md["kp"],
                md["kp"],
                md["kp"],
                md["km"],
                md["km"],
                md["km"],
                md["pip"],
                md["kp"],
                md["pim"],
                md["km"],
                md["pip"],
                md["pim"],
            ]
        )
        cov_fin = np.array(
            [
                cov_a[:, 0],
                cov_a[:, 0],
                cov_a[:, 0],
                cov_a[:, 0],
                cov_a[:, 1],
                cov_a[:, 1],
                cov_a[:, 1],
                cov_a[:, 2],
                cov_a[:, 2],
                cov_a[:, 2],
                cov_a[:, 3],
                cov_a[:, 1],
                cov_a[:, 4],
                cov_a[:, 2],
                cov_a[:, 3],
                cov_a[:, 4],
            ]
        )
    elif flav == "mu":
        cov_a = np.array([md["kl"], md["kp"], md["kp"], md["pim"], md["km"], md["pip"]])
        cov_fin = np.array(
            [
                cov_a[:, 0],
                cov_a[:, 1],
                cov_a[:, 1],
                cov_a[:, 4],
                cov_a[:, 2],
                cov_a[:, 3],
            ]
        )
    elif flav == "amu":
        cov_a = np.array([md["kl"], md["km"], md["km"], md["pip"], md["kp"], md["pim"]])
        cov_fin = np.array(
            [
                cov_a[:, 0],
                cov_a[:, 2],
                cov_a[:, 2],
                cov_a[:, 3],
                cov_a[:, 1],
                cov_a[:, 4],
            ]
        )
    elif flav == "e":
        cov_a = np.array([md["kl"], md["kp"], md["pip"], md["kp"]])
        cov_fin = np.array([cov_a[:, 0], cov_a[:, 1], cov_a[:, 3], cov_a[:, 1]])
    elif flav == "ae":
        cov_a = np.array([md["kl"], md["km"], md["pim"], md["km"]])
        cov_fin = np.array([cov_a[:, 0], cov_a[:, 2], cov_a[:, 4], cov_a[:, 2]])
    return cov_fin


def cov_flux(theta, Enu, flav, m_file):
    arr_flux = []
    if flav == "tot":
        decayid = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    elif flav == "mu":
        decayid = np.array([3, 5, 7, 12, 13])
    elif flav == "amu":
        decayid = np.array([4, 8, 10, 11, 14])
    elif flav == "e":
        decayid = np.array([1, 6, 11])
    elif flav == "ae":
        decayid = np.array([2, 9, 12])
    for i in range(len(decayid)):
        if decayid[i] in [11, 12]:
            arr_flux.append(makeflux(decayid[i], theta, Enu, True, "pi"))
            arr_flux.append(makeflux(decayid[i], theta, Enu, True, "kp"))
        else:
            arr_flux.append(makeflux(decayid[i], theta, Enu, False))

    cov_mat = get_cov_mat(m_file, flav)

    store = []
    for i in range(len(cov_mat)):
        for j in range(len(cov_mat)):
            if i >= j:
                store.append(arr_flux[i] * (cov_mat[i][j]) * arr_flux[j])
    arr_flux_err = np.sqrt(np.sum(store, axis=0))
    flux_tot = np.sum(arr_flux, axis=0)

    return arr_flux_err / flux_tot


pot = 6.6 * 1e21
nAr_nucleons = 1e30


# calculate number of events from theoretical prediction (=flux*cross section)
def makeevents(
    theta,
    pot,
    decayid,
    Rqe,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
    Enu,
    id,
):
    fluxin = makeflux(decayid, theta, Enu)
    if fluxin < 1e-15:
        flux = 0
    else:
        flux = fluxin
    numevents = (
        flux
        * xsecs(Enu, Rqe, Rres, Rdis, Rmec, gammaqel, gammares, gammadis, gammamec, id)
        * scaling_val
    )
    return numevents


def get_error(meas_arr):
    error_tot = []
    for j in range(len(meas_arr)):
        error = []
        for i in range(len(meas_arr[0][:, 1])):
            if round(meas_arr[j][:, 1][i], 3) != 0.00:
                error.append(np.sqrt(abs(meas_arr[j][:, 1][i])))
            else:
                error.append(1)
        error_tot.append(error)
    return error_tot


def make_interp_eff(proc_name, Enu):
    Enu_meas = Enu  # np.arange(0.125, 10.625,0.25)
    bin_centres = np.loadtxt("eff/glb_bin_centres.txt")
    """nue returns nuesig, nuebar_bkg, numu_bkg, numubar_bkg, nc_bkg
    nuebar returns nuebarsig, nue_bkg, numu_bkg, numubar_bkg, nc_bkg
    numu returns numusig, [nue_bkg, nuebar_bkg], nc_bkg
    numubar returns numubarsig, [nue_bkg, nuebar_bkg], nc_bkg"""
    effs = []
    file_ending = []
    if proc_name == "electron" or proc_name == "anti_electron":
        file_prefix = "post_app_FHC_"
        file_ending = ["nue_sig", "nuebar_bkg", "numu_bkg", "numubar_bkg", "NC_bkg"]
        if proc_name == "anti_electron":
            file_ending = ["nue_bkg", "nuebar_sig", "numu_bkg", "numubar_bkg", "NC_bkg"]
    if proc_name == "muon" or proc_name == "anti_muon":
        file_prefix = "post_dis_FHC_"
        file_ending = ["nue_bkg", "nue_bkg", "numu_sig", "", "NC_bkg"]
        if proc_name == "anti_muon":
            file_ending = ["nue_bkg", "nue_bkg", "", "numubar_sig", "NC_bkg"]

    for i in range(len(file_ending)):
        if file_ending[i] == "":
            effs.append(np.zeros(len(Enu_meas)))
        else:
            eff = np.loadtxt(
                "eff/" + file_prefix + file_ending[i] + ".txt",
                delimiter=",",
            )[:67]  # get efficiencies up to 11GeV
            effs.append(
                np.interp(Enu_meas, bin_centres, eff)
            )  # interpolate efficiency histogram to energies that match bins in flux hists over range that matches efficiency list, then return range which matches flux hists

    return effs


def conv_params(fit_params):
    return np.array(
        [
            fit_params[0],
            fit_params[0],
            fit_params[0],
            fit_params[0],
            fit_params[1],
            fit_params[1],
            fit_params[1],
            fit_params[2],
            fit_params[2],
            fit_params[2],
            fit_params[4],
            fit_params[5],
            fit_params[6],
            fit_params[7],
        ]
    )


def make_plot_grid_combined(p_type, fit_params, Enu, meas_arr, err_arr, m_val):
    v = m_val
    trans_vars = {
        1: v["p_kl"],
        2: v["p_kl"],
        3: v["p_kl"],
        4: v["p_kl"],
        5: v["p_kp"],
        6: v["p_kp"],
        7: v["p_kp"],
        8: v["p_km"],
        9: v["p_km"],
        10: v["p_km"],
        11: v["p_mup"],
        12: v["p_mum"],
        13: v["p_pip"],
        14: v["p_pim"],
    }
    trans_true = {
        1: fit_params[0],
        2: fit_params[0],
        3: fit_params[0],
        4: fit_params[0],
        5: fit_params[1],
        6: fit_params[1],
        7: fit_params[1],
        8: fit_params[2],
        9: fit_params[2],
        10: fit_params[2],
        11: fit_params[3],
        12: fit_params[4],
        13: fit_params[5],
        14: fit_params[6],
    }
    theta = [0.0, 0.0209, 0.03136, 0.04181, 0.05226]
    color = ["m", "b", "g", "orange", "c"]
    if p_type == "muon":
        func = [muon_noeff_0, muon_noeff_1]
        id = [3, 5, 7, 12, 13]
        id_str = "muon"
        offset = 0
        m_obj = [trans_vars[i] for i in id]
        true_pars = [trans_true[i] for i in id]
    if p_type == "a_muon":
        func = [
            anti_muon_noeff_0,
            anti_muon_noeff_1,
        ]
        id = [4, 8, 10, 11, 14]
        id_str = "anti_muon"
        offset = 5
        m_obj = [trans_vars[i] for i in id]
        true_pars = [trans_true[i] for i in id]
    if p_type == "electron":
        func = [
            electron_noeff_0,
            electron_noeff_1,
        ]
        id = [1, 6, 11]
        id_str = "electron"
        offset = 10
        m_obj = [trans_vars[i] for i in id]
        true_pars = [trans_true[i] for i in id]
    if p_type == "a_electron":
        func = [
            anti_electron_noeff_0,
            anti_electron_noeff_1,
        ]
        id = [2, 9, 12]
        id_str = "anti_electron"
        m_obj = [trans_vars[i] for i in id]
        true_pars = [trans_true[i] for i in id]
        offset = 12

    n_params = len(id)
    fig, ax = plt.subplots(5, len(id), figsize=(36, 18), sharex=True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for i in range(5):
        for k in range(0, n_params):
            initialize_fluxes(True)
            ax[i][k].plot(
                Enu,
                m_obj[k]
                * makeflux(id[k], theta[i], Enu)
                * scaling_val
                * xsecs(Enu, 0, 0, 0, 0, 0, 0, 0, 0, id_str),
                label=str(id[k]),
                linestyle="-",
                color=color[k],
                linewidth=5,
            )
            ax[i][k].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax[i][k].tick_params(
                axis="both", which="both", direction="in", labelsize=28
            )
            initialize_fluxes(False)
            ax[i][k].plot(
                Enu,
                true_pars[k]
                * makeflux(id[k], theta[i], Enu)
                * scaling_val
                * xsecs(Enu, 0, 0, 0, 0, 0, 0, 0, 0, id_str),
                linestyle="--",
                color=color[k],
                marker="x",
                linewidth=5,
                markersize=10,
            )
            # ax[i][k].legend(frameon=False)
            ax[i][k].yaxis.get_offset_text().set_fontsize(34)
            ax[4][k].set_xlabel("$E_\\nu$ [GeV]")
            ax[i][0].set_ylabel("$N_{events}$")
            if id[k] == 13 and theta[i] == 0.0:
                np.savetxt(
                    "pip_numu_data.txt",
                    np.transpose(
                        [
                            Enu,
                            true_pars[k]
                            * makeflux(id[k], theta[i], Enu)
                            * scaling_val
                            * xsecs(Enu, 0, 0, 0, 0, 0, 0, 0, 0, id_str),
                        ]
                    ),
                )

    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.xlabel("$E_\\nu$ [GeV]")

    plt.show()


def make_plot_grid_no_comb(p_type, fit_params, Enu, meas_arr, err_arr, m_obj):
    theta = [0.0, 0.0209, 0.03136, 0.04181, 0.05226]
    color = ["m", "b", "g", "orange", "c"]
    if p_type == "muon":
        func = [muon_noeff_0, muon_noeff_1]
        id = [3, 5, 7, 12, 13]
        id_str = "muon"
        offset = 0
    if p_type == "a_muon":
        func = [
            anti_muon_noeff_0,
            anti_muon_noeff_1,
        ]
        id = [4, 8, 10, 11, 14]
        id_str = "anti_muon"
        offset = 5
    if p_type == "electron":
        func = [
            electron_noeff_0,
            electron_noeff_1,
        ]
        id = [1, 6]
        id_str = "electron"
        offset = 10
    if p_type == "a_electron":
        func = [
            anti_electron_noeff_0,
            anti_electron_noeff_1,
        ]
        id = [2, 9]
        id_str = "anti_electron"
        offset = 12

    n_params = len(id)
    fig, ax = plt.subplots(5, len(id), figsize=(36, 18), sharex=True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for i in range(5):
        for k in range(0, n_params):
            ax[i][k].plot(
                Enu,
                m_obj[k + offset]
                * makeflux(id[k], theta[i], Enu)
                * scaling_val
                * xsecs(Enu, 0, 0, 0, 0, 0, 0, 0, 0, id_str),
                label=str(id[k]),
                linestyle="-",
                color=color[k],
                linewidth=5,
            )
            ax[i][k].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax[i][k].tick_params(
                axis="both", which="both", direction="in", labelsize=24
            )
            ax[i][k].yaxis.get_offset_text().set_fontsize(24)

            ax[4][k].set_xlabel("$E_\\nu$ [GeV]", fontsize=30)
            ax[i][0].set_ylabel("$N_{events}$", fontsize=30)
        ax[i][0].legend(frameon=False)

    plt.subplots_adjust(wspace=0.15, hspace=0.2)

    plt.show()


def get_contours(arr, p_names, m_object):
    for i in range(len(p_names)):
        for j in range(len(p_names)):
            if i == j:
                arr[i].append(m_object.profile(p_names[i], size=20))
            else:
                arr[i].append(m_object.mncontour(p_names[i], p_names[j], cl=1, size=20))
    return arr


def make_contour_plots(dir, ext, fit_params=[1, 1, 1, 1, 1]):
    fig, axes = plt.subplots(
        figsize=(32, 32), sharex=False, sharey=False, ncols=5, nrows=5
    )
    param_names = np.array(["p_kl", "p_kp", "p_km", "p_pip", "p_pim"])
    format_names = np.array(["$K^0_L$", "$K^+$", "$K^-$", "$\\pi^+$", "$\\pi^-$"])
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            if i < j:
                axes[i, j].axis("off")
            else:
                if i == j:
                    arr_contour = np.loadtxt(dir + param_names[j] + ext + "_fhc.txt")
                    arr_contour_rhc = np.loadtxt(
                        dir + param_names[j] + ext + "_rhc.txt"
                    )
                    x = arr_contour[0]
                    x_rhc = arr_contour_rhc[0]
                    y = arr_contour[1]
                    y_rhc = arr_contour_rhc[1]
                    axes[i, j].plot(
                        np.repeat(fit_params[i], 10),
                        np.linspace(0, np.max(y), 10),
                        linestyle="--",
                    )
                    axes[i, j].set_xlim(0.8, 1.2)
                    axes[i, j].set_ylim(0, 4)
                    axes[i, j].plot(x, y, color="blue", label="FHC")
                    axes[i, j].plot(
                        x_rhc, y_rhc, color="orange", label="RHC", linestyle=":"
                    )
                    axes[i, j].tick_params(axis="x", which="major", pad=20)
                    axes[i, j].tick_params(axis="y", which="major", pad=10)
                else:
                    arr_contour = np.transpose(
                        np.loadtxt(
                            dir
                            + param_names[j]
                            + "_"
                            + param_names[i]
                            + ext
                            + "_fhc.txt"
                        )
                    )
                    arr_contour_rhc = np.transpose(
                        np.loadtxt(
                            dir
                            + param_names[j]
                            + "_"
                            + param_names[i]
                            + ext
                            + "_rhc.txt"
                        )
                    )
                    x = arr_contour[0]
                    x_rhc = arr_contour_rhc[0]
                    y = arr_contour[1]
                    y_rhc = arr_contour_rhc[1]
                    axes[i, j].plot(
                        fit_params[i],
                        fit_params[j],
                        marker="x",
                        linestyle="",
                        markersize=20,
                        color="white",
                    )
                    axes[i, j].set_xlim(0.8, 1.2)
                    axes[i, j].set_ylim(0.8, 1.2)
                    axes[i, j].tick_params(axis="x", which="major", pad=20)
                    axes[i, j].tick_params(axis="y", which="major", pad=10)
                    axes[i, j].fill(x, y, color="blue", alpha=0.5, label="FHC")
                    axes[i, j].fill(
                        x_rhc, y_rhc, color="orange", alpha=0.5, label="RHC"
                    )

                if j == 0:
                    axes[i, j].set_ylabel(format_names[i])
                    # axes[0,j].legend()
                if i == 4:
                    axes[i, j].set_xlabel(format_names[j])
                    axes[i, 0].legend(bbox_to_anchor=(2.53, 5.9), prop={"size": 50})
                # axes[i,j].legend(frameon=False,loc='upper right')
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.savefig(
        "./triangle.pdf",
        bbox_inches="tight",
        dpi=300,
    )


def check_zero_add(flux_it, flux_tot):
    if len(flux_it) > 0:
        flux_tot += flux_it[:, 1]
    return flux_tot


# generate number of events with given ratio of contributions
def makeeventsmeas_fitparams_eff(theta, fit_params, pot_share, Ebins, id, smooth):
    # TODO raise FileNotFoundError for missing files like in offaxis_fluxes_loop
    # TODO assert matching lengths of files/params

    p_kl, p_kp, p_km, p_pip, p_pim = fit_params

    if id == "muon":
        eff = make_interp_eff("muon", Ebins)
    elif id == "anti_muon":
        eff = make_interp_eff("anti_muon", Ebins)
    elif id == "electron":
        eff = make_interp_eff("electron", Ebins)
    elif id == "anti_electron":
        eff = make_interp_eff("anti_electron", Ebins)
    else:
        raise ValueError(
            "Invalid neutrino type specified - input must be muon, anti_muon, electron, or anti_electron"
        )

    params = {
        "numu": np.array([p_kl, p_kp, p_kp, p_pip, p_pim, p_km]),
        "numubar": np.array([p_kl, p_km, p_km, p_pim, p_pip, p_kp]),
        "nue": np.array([p_kl, p_kp, p_pip, p_kp]),
        "nuebar": np.array([p_kl, p_km, p_pim, p_km]),
    }

    flux = {}
    for flux_type in ["numu", "nue", "numubar", "nuebar"]:
        files = (
            sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}.txt"))
            + sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}_pi.txt"))
            + sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}_kp.txt"))
        )
        flux[flux_type] = np.sum(
            [
                np.loadtxt(files[i])[:, 1] * params[flux_type][i]
                for i in range(len(files))
            ],
            axis=0,
        )
        if flux_type == "numu":
            Enu = np.loadtxt(files[0])[:, 0]

    flux["nue"] += (
        p_pip * np.loadtxt(f"{flux_dir}mup_to_numubar_offaxis{theta}_pi.txt")[:, 1]
        + p_kp * np.loadtxt(f"{flux_dir}mup_to_numubar_offaxis{theta}_kp.txt")[:, 1]
    )
    flux["nuebar"] += (
        p_pim * np.loadtxt(f"{flux_dir}mum_to_numu_offaxis{theta}_pi.txt")[:, 1]
        + p_km * np.loadtxt(f"{flux_dir}mum_to_numu_offaxis{theta}_kp.txt")[:, 1]
    )

    if smooth:
        flux[flux_type] = gaussian_filter(flux[flux_type], sigma=0.8)

    Rqe, Rdis, Rres, Rmec, gammaqel, gammadis, gammares, gammamec = (
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    )

    xsec = {
        "numu": xsecs(
            Enu, Rqe, Rres, Rdis, Rmec, gammaqel, gammares, gammadis, gammamec, "muon"
        ),
        "numubar": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_muon",
        ),
        "nue": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "electron",
        ),
        "nuebar": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_electron",
        ),
        "nc": xsecs(
            Enu, Rqe, Rres, Rdis, Rmec, gammaqel, gammares, gammadis, gammamec, "nc"
        ),
        "anc": xsecs(
            Enu, Rqe, Rres, Rdis, Rmec, gammaqel, gammares, gammadis, gammamec, "a_nc"
        ),
    }

    if id == "muon":
        numevents = (
            (
                flux["numu"] * xsec["numu"] * eff[2]
                + flux["nue"] * xsec["nue"] * eff[0]
                + flux["nuebar"] * xsec["nuebar"] * eff[1]
                + (flux["numu"] + flux["nue"]) * xsec["nc"] * eff[4]
                + (flux["numubar"] + flux["nuebar"]) * xsec["anc"] * eff[4]
            )
            * scaling_val
            * pot_share
        )
    elif id == "anti_muon":
        numevents = (
            (
                flux["numubar"] * xsec["numubar"] * eff[3]
                + flux["nue"] * xsec["nue"] * eff[0]
                + flux["nuebar"] * xsec["nuebar"] * eff[1]
                + (flux["numu"] + flux["nue"]) * xsec["nc"] * eff[4]
                + (flux["numubar"] + flux["nuebar"]) * xsec["anc"] * eff[4]
            )
            * scaling_val
            * pot_share
        )
    elif id == "electron":
        numevents = (
            (
                flux["nue"] * xsec["nue"] * eff[0]
                + flux["nuebar"] * xsec["nuebar"] * eff[1]
                + flux["numu"] * xsec["numu"] * eff[2]
                + flux["numubar"] * xsec["numubar"] * eff[3]
                + (flux["numu"] + flux["nue"]) * xsec["nc"] * eff[4]
                + (flux["numubar"] + flux["nuebar"]) * xsec["anc"] * eff[4]
            )
            * scaling_val
            * pot_share
        )
    elif id == "anti_electron":
        numevents = (
            (
                flux["nuebar"] * xsec["nuebar"] * eff[1]
                + flux["nue"] * xsec["nue"] * eff[0]
                + flux["numu"] * xsec["numu"] * eff[2]
                + flux["numubar"] * xsec["numubar"] * eff[3]
                + (flux["numu"] + flux["nue"]) * xsec["nc"] * eff[4]
                + (flux["numubar"] + flux["nuebar"]) * xsec["anc"] * eff[4]
            )
            * scaling_val
            * pot_share
        )

    S = build_smear_matrix(Ebins)
    numevents = np.matmul(S, numevents)

    return np.column_stack((Enu, numevents))


def makeeventsmeas_fitparams_noeff(theta, fit_params, pot_share, Ebins, id):
    p_kl, p_kp, p_km, p_pip, p_pim = fit_params
    nbins = len(Ebins)

    params = {
        "numu": np.array([p_kl, p_kp, p_kp, p_pip, p_pim, p_km]),
        "numubar": np.array([p_kl, p_km, p_km, p_pim, p_pip, p_kp]),
        "nue": np.array([p_kl, p_kp, p_pip, p_kp]),
        "nuebar": np.array([p_kl, p_km, p_pim, p_km]),
    }

    flux = {}
    for flux_type in ["numu", "nue", "numubar", "nuebar"]:
        files = (
            sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}.txt"))
            + sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}_pi.txt"))
            + sorted(glob.glob(f"{flux_dir}*_to_{flux_type}_*offaxis{theta}_kp.txt"))
        )
        flux[flux_type] = np.sum(
            [
                np.loadtxt(files[i])[:, 1] * params[flux_type][i]
                for i in range(len(files))
            ],
            axis=0,
        )
        if flux_type == "numu":
            Enu = np.loadtxt(files[0])[:, 0]

    flux["nue"] += (
        p_pip * np.loadtxt(f"{flux_dir}mup_to_numubar_offaxis{theta}_pi.txt")[:, 1]
        + p_kp * np.loadtxt(f"{flux_dir}mup_to_numubar_offaxis{theta}_kp.txt")[:, 1]
    )
    flux["nuebar"] += (
        p_pim * np.loadtxt(f"{flux_dir}mum_to_numu_offaxis{theta}_pi.txt")[:, 1]
        + p_km * np.loadtxt(f"{flux_dir}mum_to_numu_offaxis{theta}_kp.txt")[:, 1]
    )

    Rqe, Rdis, Rres, Rmec, gammaqel, gammadis, gammares, gammamec = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    xsec = {
        "numu": xsecs(
            Enu, Rqe, Rres, Rdis, Rmec, gammaqel, gammares, gammadis, gammamec, "muon"
        ),
        "numubar": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_muon",
        ),
        "nue": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "electron",
        ),
        "nuebar": xsecs(
            Enu,
            Rqe,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_electron",
        ),
    }

    if id == "muon":
        numevents = flux["numu"] * xsec["numu"] * scaling_val * pot_share
    elif id == "anti_muon":
        numevents = flux["numubar"] * xsec["numubar"] * scaling_val * pot_share
    elif id == "electron":
        numevents = flux["nue"] * xsec["nue"] * scaling_val * pot_share
    elif id == "anti_electron":
        numevents = flux["nuebar"] * xsec["nuebar"] * scaling_val * pot_share

    return np.column_stack((Enu, numevents))


class Chi2Function:  # TODO add POT parameter to muon_0, axis_0 etc (which can just be passed when calling axis_i in off_axis_nuisance_all)
    def __init__(self, x_data, y_data, y_err, model):
        self.x_data = x_data
        self.y_data = y_data
        self.y_err = y_err
        self.model = model

    def __call__(
        self,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    ):
        Rqel_err = np.sqrt(0.07**2 + 0.015**2)
        Rres_err = np.sqrt(0.08**2 + 0.015**2)
        Rdis_err = np.sqrt(0.11**2 + 0.015**2)
        Rmec_err = np.sqrt(0.08**2 + 0.015**2)
        gammaqel_err = 0.04
        gammares_err = 0.1
        gammadis_err = 0.03
        gammamec_err = 0.1

        # Calculate chi-square
        y_pred = self.model(
            self.x_data,
            p_kl,
            p_kp,
            p_km,
            p_pip,
            p_pim,
            Rqel,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
        )
        residuals = self.y_data - y_pred
        chi2 = np.sum(residuals**2 / (self.y_err**2 + (0.01 * y_pred) ** 2))
        params_norm = np.array([Rqel, Rres, Rdis, Rmec])
        params_tilt = np.array([gammaqel, gammares, gammadis, gammamec])
        errors_norm = np.array([Rqel_err, Rres_err, Rdis_err, Rmec_err])
        errors_tilt = np.array([gammaqel_err, gammares_err, gammadis_err, gammamec_err])
        prior = np.sum((params_tilt / errors_tilt) ** 2) + np.sum(
            ((np.ones_like(params_norm) - params_norm) / errors_norm) ** 2
        )

        return chi2 + prior


class addChi2:
    def __init__(self, *chi2_functions):
        """
        Initialize with any number of chi2 functions.
        :param chi2_functions: A variable number of chi2 functions to combine
        """
        self.chi2_functions = chi2_functions

    def combine(
        self,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    ):
        # Sum the individual chi2 values from each function
        total_chi2 = np.sum(
            [
                chi2_func(
                    p_kl,
                    p_kp,
                    p_km,
                    p_pip,
                    p_pim,
                    Rqel,
                    Rres,
                    Rdis,
                    Rmec,
                    gammaqel,
                    gammares,
                    gammadis,
                    gammamec,
                )
                for chi2_func in self.chi2_functions
            ]
        )
        return total_chi2


class fluxFunction:
    def __init__(self, flavour, channel, theta, eff):
        self.flavour = flavour
        self.channel = channel
        self.theta = theta
        self.eff = eff

    def __call__(
        self,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    ):
        """Takes neutrino energy spectrum + scaling parameters for each flux contribution then finds neutrino event spectrum for given flavour"""

        if self.channel + self.flavour == "electron":
            self.eff_arr = [1, 0, 0, 0, 0]
        elif self.channel + self.flavour == "anti_electron":
            self.eff_arr = [0, 1, 0, 0, 0]
        elif self.channel + self.flavour == "muon":
            self.eff_arr = [0, 0, 1, 0, 0]
        elif self.channel + self.flavour == "anti_muon":
            self.eff_arr = [0, 0, 0, 1, 0]
        if self.eff:
            self.eff_arr = make_interp_eff(self.channel + self.flavour, Enu)
        eff_val = self.eff_arr  # make dummy efficiency array so if efficiencies not wanted any non-signal contribution is multiplied by 0 but signal is multiplied by 1

        p = np.array(
            [p_kl, p_kp, p_kp, p_pip, p_kl, p_km, p_km, p_pim, p_kl, p_kp, p_kl, p_km]
        )
        id = np.array([3, 5, 7, 13, 4, 8, 10, 14, 1, 6, 2, 9])
        S = build_smear_matrix(Enu)
        flux = np.array([makeflux(i, self.theta, Enu, False) for i in id])
        xsec_mu = xsecs(
            Enu, Rqel, Rdis, Rres, Rmec, gammaqel, gammares, gammadis, gammamec, "muon"
        )
        xsec_amu = xsecs(
            Enu,
            Rqel,
            Rdis,
            Rres,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_muon",
        )
        xsec_e = xsecs(
            Enu,
            Rqel,
            Rdis,
            Rres,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "electron",
        )
        xsec_ae = xsecs(
            Enu,
            Rqel,
            Rdis,
            Rres,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
            "anti_electron",
        )
        xsec_nc = xsecs(
            Enu, Rqel, Rdis, Rres, Rmec, gammaqel, gammares, gammadis, gammamec, "nc"
        )
        xsec_anc = xsecs(
            Enu, Rqel, Rdis, Rres, Rmec, gammaqel, gammares, gammadis, gammamec, "a_nc"
        )

        tot_mu = np.sum(p[:4].reshape(-1, 1) * flux[:4] * xsec_mu * eff_val[2], axis=0)
        tot_amu = np.sum(
            p[4:8].reshape(-1, 1) * flux[4:8] * xsec_amu * eff_val[3], axis=0
        )
        tot_e = np.sum(
            p[8:10].reshape(-1, 1) * flux[8:10] * xsec_e * eff_val[0], axis=0
        )
        tot_ae = np.sum(
            p[10:].reshape(-1, 1) * flux[10:] * xsec_ae * eff_val[1], axis=0
        )
        tot_nc = np.sum(
            p[:4].reshape(-1, 1) * flux[:4] * xsec_nc * eff_val[4], axis=0
        ) + np.sum(p[8:10].reshape(-1, 1) * flux[8:10] * xsec_nc * eff_val[4], axis=0)
        tot_anc = np.sum(
            p[4:8].reshape(-1, 1) * flux[4:8] * xsec_anc * eff_val[4], axis=0
        ) + np.sum(p[10:].reshape(-1, 1) * flux[10:] * xsec_anc * eff_val[4], axis=0)

        mup_flux = p_pip * makeflux(11, self.theta, Enu, True, "pi") + p_kp * makeflux(
            11, self.theta, Enu, True, "kp"
        )
        mum_flux = p_pim * makeflux(12, self.theta, Enu, True, "pi") + p_km * makeflux(
            12, self.theta, Enu, True, "kp"
        )

        tot_mu += mum_flux * xsec_mu * eff_val[2]
        tot_nc += mum_flux * xsec_nc * eff_val[4]
        tot_nc += mup_flux * xsec_nc * eff_val[4]
        tot_amu += mup_flux * xsec_amu * eff_val[3]
        tot_anc += mup_flux * xsec_anc * eff_val[4]
        tot_anc += mum_flux * xsec_anc * eff_val[4]
        tot_e += mup_flux * xsec_e * eff_val[0]
        tot_ae += mum_flux * xsec_ae * eff_val[1]
        fin_events = (
            (tot_mu + tot_amu + tot_e + tot_ae + tot_nc + tot_anc)
            * scaling_val
            * pot_share
        )
        return np.matmul(S, fin_events)


# TODO make function which does below instead of repeating for each neutrino flavour and angle
muon_functions = [fluxFunction("muon", "", theta, True) for theta in [0.0, 0.06272]]
muon_0, muon_1 = muon_functions
anti_muon_functions = [
    fluxFunction("muon", "anti_", theta, True) for theta in [0.0, 0.06272]
]
(
    anti_muon_0,
    anti_muon_1,
) = anti_muon_functions
electron_functions = [
    fluxFunction("electron", "", theta, True) for theta in [0.0, 0.06272]
]
electron_0, electron_1 = electron_functions
anti_electron_functions = [
    fluxFunction("electron", "anti_", theta, True) for theta in [0.0, 0.06272]
]
(
    anti_electron_0,
    anti_electron_1,
) = anti_electron_functions
muon_functions_noeff = [
    fluxFunction("muon", "", theta, False) for theta in [0.0, 0.06272]
]
(
    muon_noeff_0,
    muon_noeff_1,
) = muon_functions_noeff
anti_muon_functions_noeff = [
    fluxFunction("muon", "anti_", theta, False) for theta in [0.0, 0.06272]
]
(
    anti_muon_noeff_0,
    anti_muon_noeff_1,
) = anti_muon_functions_noeff
electron_functions_noeff = [
    fluxFunction("electron", "", theta, False) for theta in [0.0, 0.06272]
]
(
    electron_noeff_0,
    electron_noeff_1,
) = electron_functions_noeff
anti_electron_functions_noeff = [
    fluxFunction("electron", "anti_", theta, False) for theta in [0.0, 0.06272]
]
(
    anti_electron_noeff_0,
    anti_electron_noeff_1,
) = anti_electron_functions_noeff


def axis_func(
    index,
    eff,
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    if eff:
        eff_str = ""
    else:
        eff_str = "noeff_"
    muon_func = globals()[f"muon_{eff_str}{index}"]
    anti_muon_func = globals()[f"anti_muon_{eff_str}{index}"]
    electron_func = globals()[f"electron_{eff_str}{index}"]
    anti_electron_func = globals()[f"anti_electron_{eff_str}{index}"]

    return (
        muon_func(
            pot_share,
            Enu,
            p_kl,
            p_kp,
            p_km,
            p_pip,
            p_pim,
            Rqel,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
        )
        + anti_muon_func(
            pot_share,
            Enu,
            p_kl,
            p_kp,
            p_km,
            p_pip,
            p_pim,
            Rqel,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
        )
        + electron_func(
            pot_share,
            Enu,
            p_kl,
            p_kp,
            p_km,
            p_pip,
            p_pim,
            Rqel,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
        )
        + anti_electron_func(
            pot_share,
            Enu,
            p_kl,
            p_kp,
            p_km,
            p_pip,
            p_pim,
            Rqel,
            Rres,
            Rdis,
            Rmec,
            gammaqel,
            gammares,
            gammadis,
            gammamec,
        )
    )


def axis_0_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        0,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_1_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        1,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_2_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        2,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_3_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        3,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_4_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        4,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_5_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        5,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_6_ne(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        6,
        False,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_0_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        0,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_1_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        1,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_2_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        2,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_3_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        3,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_4_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        4,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_5_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        5,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )


def axis_6_e(
    pot_share,
    Enu,
    p_kl,
    p_kp,
    p_km,
    p_pip,
    p_pim,
    Rqel,
    Rres,
    Rdis,
    Rmec,
    gammaqel,
    gammares,
    gammadis,
    gammamec,
):
    return axis_func(
        6,
        True,
        pot_share,
        Enu,
        p_kl,
        p_kp,
        p_km,
        p_pip,
        p_pim,
        Rqel,
        Rres,
        Rdis,
        Rmec,
        gammaqel,
        gammares,
        gammadis,
        gammamec,
    )
