import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from numpy import loadtxt
import math
from numpy.polynomial import polynomial as P
import csv
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline, splrep, splev
from scipy.ndimage import gaussian_filter
from iminuit import cost, Minuit
from iminuit.cost import LeastSquares  # , LeastSquaresMargot
from functools import partial
import timeit
import pickle

import sys

sys.path.append("/unix/dune2/mmacmahon/dune-prism/scripts")


from analysis_helper import *


from scipy.interpolate import make_interp_spline
from multiprocessing import Pool

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

time = eval(sys.argv[2])
run_contour = eval(sys.argv[3])
prefix = "/unix/dune2/mmacmahon/dune-prism/scripts/qgsp_outputs"
run_n = "all_on_qgsp_"
ext = str(sys.argv[1])
pkl_str = (
    prefix
    + "/fits/fit"
    + smear_str
    + param_str
    + step_str
    + "bias_"
    + run_n
    + ext
    + ".pkl"
)
flux_dir = "/unix/dune2/mmacmahon/qgsp/rhc/40_angles_80_repeats/histtot/10_GeV/"  # /unix/dune2/mmacmahon/qgsp/40_angles_80_repeats/histtot/10_GeV/'#'/unix/dune2/mmacmahon/250_repeats/histtot/'
# flux_dir_ftfp = '/unix/dune2/mmacmahon/ftfp/6_angles_320_repeats_extra_nue/histtot/10_GeV/'


# ========== Change angles used in lines 1019,1022,1025 and 1028 if changing the values of theta!! ========= #
theta = [0.0, 0.010976, 0.021952, 0.032928, 0.042336, 0.053312, 0.06272]
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
if time:
    POT = float(sys.argv[1]) * 1.1e21
else:
    POT = 1.1e21 * 6  # 6 years
print(type(POT))

initialize_xsecs(param_xsec)
initialize_fluxes(smear)
initialize_scaling(POT)
initialize_flux_dir(flux_dir)

Ebins = np.arange(0.125, 10.625, 0.25)
fit_params = np.round(np.random.uniform(low=1.0, high=1.0, size=(5,)), 2)
p_kl, p_kp, p_km, p_pip, p_pim = fit_params

# set share of POT time each off-axis position will recieve

pot_shares = [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]  # even split
# pot_shares = [0.15,0.15,0.7/5,0.7/5,0.7/5,0.7/5,0.7/5] #30% of time in on axis/1 pos off, rest split over off axis
# pot_shares = [0.5/6,0.5/6,0.5/6,0.5/6,0.5/6,0.5/6,0.5] #half of time max off axis, the rest split evenly

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

# meas=makeeventsmeas(filein,POT)
flu_dec = []
for i in range(len(theta)):
    # filein=np.loadtxt("file_out_Neutrino_doublestats/histtot_fitparams/histtot_all_offaxis"+str(theta[i])+"_"+str(Ebins)+"Ebins_"+rat_string+".txt")
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
print(len(Enu))

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


# initialize_flux_dir(flux_dir_ftfp)
# make total axis distribution functions with correct pot share
axes = [partial(globals()[f"axis_{i}_e"], pot_shares[i]) for i in range(7)]
axis_0, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6 = axes

chi2_ax_0 = Chi2Function(
    x_data=Enu, y_data=meas_axis[0], y_err=meas_axis_err[0], model=axis_0
)
chi2_ax_1 = Chi2Function(
    x_data=Enu, y_data=meas_axis[1], y_err=meas_axis_err[1], model=axis_1
)
chi2_ax_2 = Chi2Function(
    x_data=Enu, y_data=meas_axis[2], y_err=meas_axis_err[2], model=axis_2
)
chi2_ax_3 = Chi2Function(
    x_data=Enu, y_data=meas_axis[3], y_err=meas_axis_err[3], model=axis_3
)
chi2_ax_4 = Chi2Function(
    x_data=Enu, y_data=meas_axis[4], y_err=meas_axis_err[4], model=axis_4
)
chi2_ax_5 = Chi2Function(
    x_data=Enu, y_data=meas_axis[5], y_err=meas_axis_err[5], model=axis_5
)
chi2_ax_6 = Chi2Function(
    x_data=Enu, y_data=meas_axis[6], y_err=meas_axis_err[6], model=axis_6
)

comb_chi2 = addChi2(
    chi2_ax_0, chi2_ax_1, chi2_ax_2, chi2_ax_3, chi2_ax_4, chi2_ax_5, chi2_ax_6
)

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
]  # ,False,False]#,False,False,False,False,False, False,False]

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
]  # ,(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf)]#,(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2)]
# m.errors = 1
m.simplex()
m.migrad()  # run the minimization
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

    def compute_contour(args):
        arr, param_names, m = args
        i = arr[0]
        j = arr[1]
        if i == j:
            t = m.mnprofile(param_names[i], size=20)
            s = param_names[i]
            return (np.array([t[0], t[1]]), s)
        elif i < j:
            t = m.mncontour(
                param_names[i], param_names[j], cl=1, size=20, interpolated=200
            )
            s = param_names[i] + "_" + param_names[j]
            return (t, s)
        else:
            return None

    param_indices = [
        (np.array([i, j]), param_names, m)
        for i in range(len(param_names[:5]))
        for j in range(len(param_names[:5]))
        if i <= j
    ]

    with Pool() as pool:
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
