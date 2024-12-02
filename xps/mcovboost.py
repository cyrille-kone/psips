import json
import numpy as np
from pathlib import Path
import seaborn as sns
from ext.lbd import cpt_lbd_correl
import matplotlib.pyplot as plt
from ext.plot_figs import group_boxp, save_fig, __NAMES__
w_elise = np.array([0.0077, 0.0016, 0.0007, 0.023, 0.0048, 0.00066, 0.00079, 0.018, 0.14, 0.0011, 0.0014, 0.021, 0.0089, 0.025, 0.35, 0.0014, 0.0015, 0.013, 0.38, 0.0012])
arms_means = np.array([[9.50479943, 6.85646198, 4.56226268],
                       [9.29302574, 6.64118217, 4.03600899],
                       [9.05368656, 6.40687999, 3.56388296],
                       [10.21251518, 7.48941208, 4.42843301],
                       [10.04680837, 7.19967835, 4.36182393],
                       [8.34379173, 5.66642669, 3.51154544],
                       [8.22174773, 5.45532112, 3.64021428],
                       [9.74560492, 7.2730926, 4.7095302],
                       [10.42726889, 7.61035762, 4.71849887],
                       [8.93761259, 6.18826412, 3.84374416],
                       [7.80669637, 5.26269019, 3.97029191],
                       [8.85008761, 6.58892648, 4.7335634],
                       [8.4411757, 6.15273269, 4.5890408],
                       [9.92900909, 7.39079852, 4.74927053],
                       [9.68315255, 7.20340552, 4.91191932],
                       [7.51479976, 5.3082677, 3.95508249],
                       [7.26542972, 4.99043259, 4.01638302],
                       [8.61558951, 6.33327963, 4.66343909],
                       [10.34531673, 7.769801, 5.00327494],
                       [8.29304914, 5.92157842, 3.86702564]])
covariance = np.diag(np.array([0.70437039, 0.82845749, 1.53743137]))
compute_w_star = False
savefig = True
__fname__ = "" #"mcovboost_d001_g5k"
T_star, (w_star, br) = 2208.627161419921, [None,]*2
niter = 200 # number  of FW iterations
if __name__ == "__main__":
    # load the file
    if compute_w_star:
        w_star, T_star, br = cpt_lbd_correl(arms_means, covariance, niter=niter, return_br=True)
    data = []
    delta = None
    with open(Path() / f"{__fname__}.json") as f:
        parsed = json.load(f)
        delta = parsed["meta"]["delta"]
        print(delta)
        for key, val in parsed["algo"].items():
            data += [(key, np.array(val["result"]), val["duration"])]
    print(*[(data[i][0], data[i][1].mean(0)) for i in range(len(data))])
    fig = group_boxp(*[(__NAMES__[name], result[:, 1]) for (name, result, _) in data])
    if T_star is not None:
        fig.axes[0].axhline(T_star*np.log(1/delta), color=sns.color_palette("deep")[-3], label=r"$T^*(\bm{\theta})\log({1}/{\delta})$", mfc='w', linewidth=1.5)
        fig.axes[0].legend()
    if savefig:
        save_fig(fig, f"{__fname__}.pdf")
    fig.show()