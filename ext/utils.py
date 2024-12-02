# @title Install
# %load_ext cython
import abc
import numpy as np

inf = (1 << 31) * 1.


def is_non_dominated(Y: np.ndarray, eps=0.) -> np.ndarray:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
        :param eps:
    """
    Y1 = np.expand_dims(Y, -3)
    Y2 = np.expand_dims(Y, -2)
    dominates = (Y1 >= Y2 + eps).all(axis=-1) & (Y1 > Y2 + eps).any(axis=-1)
    nd_mask = ~(dominates.any(axis=-1))
    return nd_mask


# @title  Set up
def batch_multivariate_normal(batch_mean, batch_cov) -> np.ndarray:
    r"""Batch samples from a multivariate normal
    Parameters
    ----------
    batch_mean: np.ndarray of shape [N, d]
                Batch of multivariate normal means
    batch_cov: np.ndarray of shape [N, d, d]
                Batch of multivariate normal covariances
    Returns
    -------
    Samples from N(batch_mean, batch_cov)"""
    batch_size = np.shape(batch_mean)[0]
    samples = np.arange(batch_size).astype(np.float32).reshape(-1, 1)
    return np.apply_along_axis(
        lambda i: np.random.multivariate_normal(mean=batch_mean[int(i[0])], cov=batch_cov),
        axis=1,
        arr=samples)


class Bandit(object):
    r"""Base class for bandit sampler"""

    def __init__(self, arms_means):
        self.arms_means = np.asarray(arms_means)
        self.K = len(arms_means)
        self.arms_space = np.arange(self.K)
        self.D = np.shape(arms_means)[-1]
        self.ps_mask = is_non_dominated(self.arms_means)
        self.ps = self.arms_space[self.ps_mask]

    @abc.abstractmethod
    def sample(self, arms):
        r"""Get batch samples form arms"""
        raise NotImplementedError
    def initialize(self):
        r""" Re-initialize the bandit environment"""
    @property
    def subg_mat(self):
        return ValueError()


class GaussianBandit(Bandit):
    r"""Implement a Gaussian bandit"""
    def __init__(self, arms_means, cov=None) -> None:
        r"""
        @constructor
        Parameters
        ----------
        K: int > 0
           Number of arms of the bandit
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm
        arms_scale: float or np.ndarray
                   scale or covariance matrix of each arm
        D: int>0
           Dimension of the reward vector
        """
        super(GaussianBandit, self).__init__(arms_means)
        self._arms_means = self.arms_means.reshape(-1, self.D).squeeze(-1) if self.D == 1 else self.arms_means
        self.cov = np.eye(self.D) if cov is None else cov


    def sample(self, arms):
        r"""
        Sample from a Gaussiant bandit
        Parameters
        -----------
        arms : set  of arms to sample
        Returns
        ------
        Samples from arms
        Test
        ----
        >>> gaussian_bandit= GaussianBandit(K=10)
        >>> gaussian_bandit.sample([1,2,4])
        """
        arms = [arms] if isinstance(arms, int) else np.asarray(arms, dtype=int)
        if self.D > 1:
            return batch_multivariate_normal(self._arms_means[arms], self.cov)
        elif self.D == 1:
            return np.random.normal(loc=self._arms_means[arms], scale=np.sqrt(self.cov)).reshape(-1, 1)
        raise ValueError(f"Value of D should be larger than or equal to 1 but given {self.D}")

    def subg_mat(self):
        return self.cov

class BernoulliBandit(Bandit):
    r"""Implement a Bernoulli bandit"""

    def __init__(self, arms_means) -> None:
        r"""
        @constructor
        Parameters
        ----------
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm """
        super(BernoulliBandit, self).__init__(arms_means)
        self._arms_means = self.arms_means.reshape(-1, self.D).squeeze(-1) if self.D == 1 else self.arms_means


    def sample(self, arms):
        r"""
         Sample from a Bernoulli bandit
         Parameters
         -----------
         arms : set  of arms to sample
         Returns
         ------
         Samples from arms
         Test
         ----
         >>> bernoulli_bandit = BernoulliBandit(K=10)
         >>> bernoulli_bandit.sample([1,2,4])
         """
        arms = [arms] if isinstance(arms, int) else arms
        return np.random.binomial(1, self.arms_means[arms]).reshape(-1, self.D)
    def subg_mat(self):
        return np.eye(self.D) * 1/4

def M(xi, xj):
    return np.max(xi - xj, -1)


def m(xi, xj):
    return np.min(xj - xi, -1)


def delta_i_plus(i, S_star, means):
    return min([M(means[i], means[j]) + inf * (j == i) for j in S_star])


def delta_i_minus(i, S_star_comp, means):
    if len(S_star_comp) == 0: return inf
    return min([max(M(means[j], means[i]), 0) + max(m(means[j], means)) for j in S_star_comp])


def Delta_i_star(i, means):
    r"""Sub-optimality gap of suboptimal arms"""
    return np.max(m(means[i], means))


def Delta_i(i, S_star, S_star_comp, means):
    r""" Sub-optimality gap """
    if i in S_star: return min(delta_i_plus(i, S_star, means), delta_i_minus(i, S_star_comp, means))
    return Delta_i_star(i, means)


def beta(T_i, delta):
    r"""confidence bonuses"""
    # this empirically tuned confidence  bonus ensures empirical correctness compared to the other algorithms of the
    # benchmark
    return np.sqrt((2.*np.log(1 / delta) +  2*np.log(1+ np.log(T_i)) + max(3*np.log(np.log(1 / delta)), 0) ) / T_i)


def beta_ij(T_i, T_j, delta):
    return beta(T_i, delta) + beta(T_j, delta)
    return np.sqrt(
        2 * (
                (np.log(1 / delta)  + np.log(np.log(1 / delta))) + 2 * np.log(1 + np.log(T_i)) + 2 * np.log(
            1 + np.log(T_j))
        ) * (
                (1 / T_i) + (1 / T_j)
        )
    )


def kl_bern(x, y):
    return x * np.log((x / (y + 1e-10)) + 1e-10) + (1 - x) * np.log((1 - x + 1e-10) / (1 - y + 1e-10))


COVBOOST_MEANS = np.array([[9.50479943, 6.85646198, 4.56226268],
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
COVBOOST_COV = np.diag(np.array([0.70437039, 0.82845749, 1.53743137]))
COVBOOST_BANDIT = GaussianBandit(COVBOOST_MEANS, COVBOOST_COV)