import matplotlib.pyplot as plt

RES_TOL_DX = 1e-6 # This could be a global variable
RES_TOL_G = 1e-6 # This could be a global variable
RES_TOL = 1e-6
TOL_EIG = 1e-5
U_SIM_SIZE=100
degree_polynomial = 2
U1_MIN=0.05
U2_MIN=0.10
U1_MAX=1.00
U2_MAX=1.00

# plt.rcParams.update({
#     "font.size": 14,
#     "axes.titlesize": 18,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
# })


plt.rcParams.update({
    "figure.titlesize": 14,#11, #14 for the appendix
    "axes.titlesize": 12,#18,   #12 for the appendic
    "axes.labelsize":14,# 16,    #14 for the appendic
    "xtick.labelsize":11,# 14,    #11 for the appendic
    "ytick.labelsize":11,# 14,     #11 for the appendic
    "legend.fontsize": 12, #14
    "font.size": 14,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})