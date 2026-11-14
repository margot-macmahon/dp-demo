import numpy as np
import os
import argparse

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

FILE_EXT = "NDLAr_energy_smallND2.txt"


def make_hists(in_dir, out_dir, n_repeats, is_fd=False, is_lots=False, bins_125=False):
    h = 300
    w = 350
    theta = np.array([0.0, 0.06272])
    if is_fd:
        h = 2 * 1900 * 1e2
        w = 1800 * 1e2
        theta = np.array([0.0])  # ,0.01045,0.0209,0.03136,0.04181,0.05226,0.06272])
    if is_lots:
        theta = np.linspace(0.0, 0.06272, 41)
    decayID = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    bins = np.arange(0, 10.625, 0.25)
    out_path = f"{out_dir}/histtot/10_GeV/"
    in_path = f"{in_dir}"
    if bins_125:
        bins = np.arange(0, 125.26, 0.25)
        out_path = f"{out_dir}/histtot/125_GeV/"

    for i in theta:
        for j in decayID:
            in_file = f"{in_path}/thetaoa{str(i)}_decayid{str(j)}"
            if j == 11 or j == 12:
                pi_file = f"{in_file}_pi_"
                kp_file = f"{in_file}_kp_"
                for p in [pi_file, kp_file]:
                    if not os.path.isfile(p + "NDLAr_energy_smallND2.txt"):
                        raise Warning(f"File {p}NDLAr_energy_smallND2.txt not found")
                    else:
                        energy = np.loadtxt(
                            f"{p}NDLAr_energy_smallND2.txt", delimiter=","
                        )
                        weights = np.loadtxt(
                            f"{p}NDLAr_weights_smallND2.txt", delimiter=","
                        )

                    hist, bin_edges = np.histogram(
                        energy,
                        bins=bins,
                        weights=weights / (0.25 * 25 * 1e5 * h * w * n_repeats),
                        density=False,
                    )  # normalise by bin width, number of flux files, no of events in file, area (in cm^2) and number of repeats
                    histop = []
                    for k in range(1, len(hist) + 1):
                        bin_centre = bin_edges[k - 1] + (
                            0.5 * (bin_edges[k] - bin_edges[k - 1])
                        )
                        histop.append([round(bin_centre, 3), hist[k - 1]])
                    hist_arr = np.array(histop)

                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    daughter_str = "_pi"
                    if "kp" in p:
                        daughter_str = "_kp"
                    np.savetxt(
                        out_path
                        + proc_names[j - 1]
                        + "_offaxis"
                        + str(i)
                        + daughter_str
                        + ".txt",
                        hist_arr,
                    )

            else:
                path = f"{in_file}_NDLAr_energy_smallND2.txt"
                if not os.path.isfile(path):
                    print("File" + path + " not found")
                    continue
                else:
                    energy = np.loadtxt(
                        f"{in_file}_NDLAr_energy_smallND2.txt",
                        delimiter=",",
                    )
                    weights = np.loadtxt(
                        f"{in_file}_NDLAr_weights_smallND2.txt",
                        delimiter=",",
                    )

                    hist, bin_edges = np.histogram(
                        energy,
                        bins=bins,
                        weights=weights / (0.25 * 25 * 1e5 * h * w * n_repeats),
                        density=False,
                    )  # normalise by bin width, number of flux files, no of events in file, area (in cm^2) and number of repeats
                    histop = []
                    for k in range(1, len(hist) + 1):
                        bin_centre = bin_edges[k - 1] + (
                            0.5 * (bin_edges[k] - bin_edges[k - 1])
                        )
                        histop.append([round(bin_centre, 3), hist[k - 1]])
                    hist_arr = np.array(histop)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    np.savetxt(
                        out_path + proc_names[j - 1] + "_offaxis" + str(i) + ".txt",
                        hist_arr,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make histograms of ND flux lists using energy and weights"
    )
    parser.add_argument("--in_dir", type=str, help="Directory with flux lists")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--n_repeats",
        type=int,
        help="Number of redecay repeats used when making the lists to ensure proper weighting",
    )
    parser.add_argument(
        "--fd",
        action="store_true",
        help="Whether fluxes have been made using far detector geometry",
    )
    parser.add_argument(
        "--lots",
        action="store_true",
        help="Whether ultra granular distribution of off axis angles has been used to make fluxes",
    )
    parser.add_argument(
        "--bins-125",
        action="store_true",
        dest="bins_125",
        help="Wheter to use binning range of 0-125 GeV instead of 0-10 GeV",
    )
    args = parser.parse_args()
    make_hists(
        args.in_dir,
        args.out_dir,
        args.n_repeats,
        is_fd=args.fd,
        is_lots=args.lots,
        bins_125=args.bins_125,
    )
