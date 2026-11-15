import numpy as np
from iminuit import Minuit
from functools import partial

import pickle

import sys
from analysis_helper import (
    makeeventsmeas_fitparams_eff,
    initialize_flux_dir,
    initialize_fluxes,
    initialize_scaling,
    initialize_xsecs,
    get_error,
    axis_0_e,
    axis_1_e,
    makepolynomfitfluxes_base,
    Chi2Function,
    addChi2,
)
from multiprocessing import Pool


def compute_contour(args):
    arr, param_names, m = args
    i = arr[0]
    j = arr[1]
    if i == j:
        t = m.mnprofile(param_names[i], size=20)
        s = param_names[i]
        return (np.array([t[0], t[1]]), s)
    elif i < j:
        t = m.mncontour(param_names[i], param_names[j], cl=1, size=20, interpolated=200)
        s = param_names[i] + "_" + param_names[j]
        return (t, s)
    else:
        return None


def _init_worker(smear, param_xsec, POT, flux_dir):
    initialize_xsecs(param_xsec)
    initialize_fluxes(smear)
    initialize_scaling(POT)
    initialize_flux_dir(flux_dir)


###-------------set smear, step and xsec parameterisation bools -----------------###

smear = False
param_xsec = True
step = False

###------------------------------------------------------------------------------###


smear_str = "no_smear_"
param_str = "true_xsec_"
step_str = ""
if smear:
    smear_str = "_smear_"
if param_xsec:
    param_str = "param_xsec_"
if step:
    step_str = "steps"

prefix = "fit_outputs"
run_n = "all_on_qgsp_"
flux_dir = "flux_histos/histtot/10_GeV/"


# ========== Change angles used in lines 1019,1022,1025 and 1028 if changing the values of theta!! ========= #
theta = [0.0, 0.06272]
# theta = [0.001568, 0.010976, 0.021952, 0.032928, 0.042336, 0.053312,0.061152]
ee = []
meas_mu = []
meas_amu = []
meas_e = []
meas_ae = []

massLAr = 67 * 1e3  # tonne?, fiducial mass of LAr detector
# dim 5x7x3, V=105 m^3
# LAr density 1417 kg/m^3
numArnucleons = 1 * 1e30
POT = 1.1e21 * 6  # 6 years
Ebins = np.arange(0.125, 10.625, 0.25)

if __name__ == "__main__":
    ext = str(sys.argv[1])
    run_contour = eval(sys.argv[2])
    pkl_str = f"{prefix}/fits/fit{smear_str}{param_str}{step_str}bias_{run_n}{ext}.pkl"

    initialize_xsecs(param_xsec)
    initialize_fluxes(smear)
    initialize_scaling(POT)
    initialize_flux_dir(flux_dir)

    fit_params = np.round(np.random.uniform(low=0.98, high=1.02, size=(5,)), 2)
    p_kl, p_kp, p_km, p_pip, p_pim = fit_params

    # set share of POT time each off-axis position will recieve
    pot_shares = [1 / 2, 1 / 2]  # even split

    assert np.allclose(sum(pot_shares), 1, rtol=0.0000001), "POT shares don't sum to 1"

    fit_params_chi = np.concatenate([fit_params, np.array([1, 1, 1, 1, 0, 0, 0, 0])])

    print(
        "p_kl = ",
        p_kl,
        "\n",
        "p_kp = ",
        p_kp,
        "\n",
        "p_km = ",
        p_km,
        "\n",
        "p_piplus = ",
        p_pip,
        "\n",
        "p_piminus = ",
        p_pim,
    )

    flu_dec = []
    for i in range(len(theta)):
        measi = []
        measi_err = []
        measi_muon = makeeventsmeas_fitparams_eff(
            theta[i], fit_params, pot_shares[i], Ebins, "muon", False
        )
        measi_anti_muon = makeeventsmeas_fitparams_eff(
            theta[i], fit_params, pot_shares[i], Ebins, "anti_muon", False
        )
        measi_electron = makeeventsmeas_fitparams_eff(
            theta[i], fit_params, pot_shares[i], Ebins, "electron", False
        )
        measi_anti_electron = makeeventsmeas_fitparams_eff(
            theta[i], fit_params, pot_shares[i], Ebins, "anti_electron", False
        )

        meas_mu.append(measi_muon)
        meas_amu.append(measi_anti_muon)
        meas_e.append(measi_electron)
        meas_ae.append(measi_anti_electron)

    meas_mu = np.array(meas_mu)
    meas_amu = np.array(meas_amu)
    meas_e = np.array(meas_e)
    meas_ae = np.array(meas_ae)

    Enu = np.array(meas_mu[0][:, 0])

    err_mu = get_error(meas_mu)
    err_amu = get_error(meas_amu)
    err_e = get_error(meas_e)
    err_ae = get_error(meas_ae)

    meas_axis = []
    meas_axis_err = []
    for i in range(len(theta)):
        meas_axis.append(
            meas_mu[i][:, 1] + meas_amu[i][:, 1] + meas_e[i][:, 1] + meas_ae[i][:, 1]
        )
        meas_axis_err.append(
            np.sqrt(
                np.array(err_mu[i]) ** 2
                + np.array(err_amu[i]) ** 2
                + np.array(err_e[i]) ** 2
                + np.array(err_ae[i]) ** 2
            )
        )

    # make total axis distribution functions with correct pot share
    axes = [partial(globals()[f"axis_{i}_e"], pot_shares[i]) for i in range(len(theta))]
    axis_0, axis_1 = axes

    chi2_ax_0 = Chi2Function(
        x_data=Enu, y_data=meas_axis[0], y_err=meas_axis_err[0], model=axis_0
    )
    chi2_ax_1 = Chi2Function(
        x_data=Enu, y_data=meas_axis[1], y_err=meas_axis_err[1], model=axis_1
    )

    comb_chi2 = addChi2(chi2_ax_0, chi2_ax_1)

    # set whether any parameters have fixed values
    m_fixed = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    m = Minuit(
        comb_chi2.combine,
        p_kl=1,
        p_kp=1,
        p_km=1,
        p_pip=1,
        p_pim=1,
        Rqel=1,
        Rres=1,
        Rdis=1,
        Rmec=1,
        gammaqel=0,
        gammares=0,
        gammadis=0,
        gammamec=0,
    )
    m.fixed = m_fixed
    m.errordef = Minuit.LEAST_SQUARES

    # set upper and lower limits for each parameter. We require that the parent particle normalisation values be constant.
    m.limits = [
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
    ]
    # run the minimization - first using the simplex algorithm, then the migrad algorithm
    m.simplex()
    m.migrad()
    # get hesse errors
    m.hesse()
    print("finished fit, running contour")

    param_names = [
        "p_kl",
        "p_kp",
        "p_km",
        "p_pip",
        "p_pim",
        "rqel",
        "rres",
        "rdis",
        "rmec",
        "gammaqel",
        "gammares",
        "gammadis",
        "gammamec",
    ]

    if run_contour:
        param_indices = [
            (np.array([i, j]), param_names, m)
            for i in range(len(param_names[:5]))
            for j in range(len(param_names[:5]))
            if i <= j
        ]

        with Pool(
            initializer=_init_worker,
            initargs=(smear, param_xsec, POT, flux_dir),
        ) as pool:
            results = pool.map(compute_contour, param_indices)

        for i in results:
            np.savetxt(prefix + "/contours/" + i[1] + ext + ".txt", i[0])

    print("finished contour")
    with open(pkl_str, "wb") as f:
        pickle.dump(m, f)

    param_names = [
        "p_kl",
        "p_kp",
        "p_km",
        "p_pip",
        "p_pim",
        "rqel",
        "rres",
        "rdis",
        "rmec",
        "gammaqel",
        "gammares",
        "gammadis",
        "gammamec",
    ]

    param_val = []
    err = []
    fp = open(
        prefix
        + "/results/out"
        + smear_str
        + param_str
        + step_str
        + "bias_"
        + run_n
        + ext
        + ".txt",
        "w",
    )
    for param in range(len(param_names)):
        param_val.append(m.values[param])
        err.append(m.errors[param])
        fp.write(
            f"{param_names[param]} - {fit_params_chi[param]}: {m.values[param]} ± {m.errors[param]} \n"
        )
    fp.write(f"Final Chi2: {m.fval} \n")
    fp.write(f"Calc Chi2: {comb_chi2.combine(*fit_params_chi)} \n")
    fp.close()

    np.savetxt(
        prefix
        + "/params/params"
        + smear_str
        + param_str
        + step_str
        + "bias_"
        + run_n
        + ext
        + ".txt",
        np.transpose([param_val, err]),
    )
