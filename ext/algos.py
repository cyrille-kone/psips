import numpy as np
from .utils import beta_ij, m, M
from joblib import Parallel, delayed
from .lbd import cpt_lb_ind, cpt_lbd_correl
from .track_and_stop import track_and_stop as tns


# batch tns using the algorithm of Crepon et al (2024)
def batch_tns(𝝻, seeds, *, delta=0.1, ncpu=-1, verbose=1, silent=True, tracking="C", cov=None, is_bern=False,
              re_norm=False):
    wrapper = lambda i: tns(𝝻, np.log(1 / delta), tracking, 100, silent, False, seeds[i], cov, is_bern, re_norm)[1:]
    return np.array(Parallel(n_jobs=ncpu, verbose=verbose)(
        delayed(wrapper)(i) for i in range(len(seeds))))


# compute the sample complexity for a batch of cov matrices
def cpt_tau_tns_batch_cov(𝝻, covs, seeds, *, delta=0.1, verbose=0, silent=True, tracking="C", ncpu=-1):
    res = np.empty((len(covs), len(seeds), 2), float)
    for i in range(len(covs)):
        res[i] = batch_tns(𝝻, seeds, verbose=verbose, delta=delta, cov=covs[i], ncpu=ncpu, silent=silent,
                           tracking=tracking, is_bern=False, re_norm=False)
    return res


def cpt_tau_tns_batch_means(𝝻s, cov, seed, *, delta=0.1, verbose=0, silent=True, ncpu=-1, tracking="C", is_bern=False,
                            re_norm=False):
    wrapper = lambda i: (
    *tns(𝝻s[i], np.log(1. / delta), tracking, 100, silent, False, seed, cov, is_bern, re_norm)[1:], i)
    res = Parallel(n_jobs=ncpu, verbose=verbose)(
        delayed(wrapper)(i) for i in range(len(𝝻s)))
    res = np.array(sorted(res, key=lambda x: x[-1]))
    return res[:, :-1]


# compute theoretical T_star for an instance and a batch of covs
def cpt_T_star_batch_cov(arms_means, covs, *, niter=100, nniter=100, verbose=0):
    nins = len(covs)
    wrp_lbd_ind = lambda i: (cpt_lb_ind(arms_means, niter=niter)[1], i)
    wrp_lbd_correl = lambda i: (cpt_lbd_correl(arms_means, covs[i], niter=nniter)[1], i)
    res_ind = Parallel(n_jobs=-1, verbose=verbose)(delayed(wrp_lbd_ind)(i) for i in range(nins))
    res_correl = Parallel(n_jobs=-1, verbose=verbose)(delayed(wrp_lbd_correl)(i) for i in range(nins))
    # resynchronize the results
    res_ind = np.array(sorted(res_ind, key=lambda x: x[-1]))
    res_correl = np.array(sorted(res_correl, key=lambda x: x[-1]))
    return res_ind[:, 0], res_correl[:, 0]


class Policy(object):
    def __init__(self, bandit):
        self.bandit = bandit
        self.K = self.bandit.K
        self.D = self.bandit.D
        self.arms_space = self.bandit.arms_space
        self.subg_mat = bandit.subg_mat()


class Auer(Policy):
    def __init__(self, bandit):
        super(Auer, self).__init__(bandit)

    def loop(self, seed: int, delta=0.1):
        np.random.seed(seed)
        K = self.bandit.K
        D = self.bandit.D
        A_1 = np.arange(K)
        total = np.zeros((K, D))
        Nc = np.zeros(K, dtype=int)
        optimal_arms = []
        while len(A_1) > 0:
            total[A_1] += self.bandit.sample(A_1) / np.sqrt(np.diag(self.bandit.subg_mat()))
            Nc[A_1] += 1
            mus = total / Nc[:, None]
            A_1 = [i for i in A_1 if np.all([m(mus[i], mus[j]) <= beta_ij(Nc[i], Nc[j], delta) for j in A_1])]
            P_1 = [i for i in A_1 if np.all([M(mus[i], mus[j]) >= beta_ij(Nc[i], Nc[j], delta) for j in A_1 if j != i])]
            A_1_notP_1 = [i for i in A_1 if not i in P_1]
            P_2 = [j for j in P_1 if
                   not np.any([M(mus[i], mus[j]) <= beta_ij(Nc[i], Nc[j], delta) for i in A_1_notP_1])]
            A_1_notP_2 = [i for i in A_1 if not i in P_2]
            optimal_arms.extend(P_2)
            A_1 = A_1_notP_2
        return set(self.bandit.ps) == set(optimal_arms), Nc.sum()

def batch_auer(bandit,seeds, *, ncpu=-1, delta=0.1, verbose=0):
    auer_alg = Auer(bandit)
    return np.array(Parallel(n_jobs=ncpu, verbose=verbose)(
        delayed(auer_alg.loop)(seed, delta) for seed in seeds))