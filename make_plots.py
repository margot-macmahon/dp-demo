import matplotlib.pyplot as plt
from analysis_helper import make_contour_plots

if __name__ == "__main__":
    plt.style.use("sty.mplstyle")
    print("making plots")
    make_contour_plots("contours/", "smear_uncorr_contour")
    print("made plots")
