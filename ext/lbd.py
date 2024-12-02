import sys
import argparse
import numpy as np
import cvxpy as cp
from .pareto_2d import PC2d
from .pareto_nd import PCnd
from itertools import product
from qpsolvers import solve_qp
from .utils import is_non_dominated

__doc__ = r''' Compute $T^\star$, $w^\star$ and the best response for PSI by 
               solving QP problems in $\bR^d$ and $\bR^p$ '''
def cpt_inf_non_ps(𝝻, 𝝨, ps, non_ps, d_vecs, w):
    r'''
    Compute the best response and $T_2^\star(\theta, w)^{-1}$ at a given $w$ when
    we add an arm to the ps
    :param 𝝻: arms_means
    :param 𝝨: covariance matrix
    :param ps: pareto set
    :param non_ps: suboptimal set
    :param d_vecs:  [d]^p
    :param w: allocation vector
    :return: T_2^\star(\theta, w)^{-1}, the best response, (arm to add to the ps, the best response of d_vec)
    '''
    # p is the size of the ps and we assume ps = {1, ..., p} and we remap
    # d_vec is a vector of size $p$ : the size of the pareto set
    # resolve the inf problem for a given w
    # E_i'are the vectors  of the canonical basis of R^p
    # e_i's are the vectors of the basis ofe R^d
    p = len(ps)
    E = np.eye(p)
    e = np.eye(𝝻.shape[-1])
    perm_ps = [ps[j] for j in range(p)]

    def solve_dual_non_ps(i, u):
        r''' solve the dual problem in $\bR^p$'''
        # arm $i$ is not in the ps
        # u a vector  of [d]^p fixed element of dvecs
        y_i = np.array([(𝝻[perm_ps[j]] - 𝝻[i]) @ e[u[j]] for j in range(p)])  # check access
        M_u = np.sum([np.outer(e[u[j]], E[j]) for j in range(p)], 0)  # check access
        C_u = [np.outer(E[j], e[u[j]]) @ 𝝨 @ np.outer(e[u[j]], E[j]) for j in range(p)]  # check access
        𝝨_i = M_u.T @ 𝝨 @ M_u / w[i]  # check access
        𝝨_ps = np.sum([C_u[j] / w[perm_ps[j]] for j in range(p)], 0)  # check access
        #x = cp.Variable(p)
        #constraints = [x >= 0]
        #objective = cp.Minimize((1. / 2) * cp.quad_form(x, 𝝨_i + 𝝨_ps, True) - x.T @ y_i)
        #prob = cp.Problem(objective, constraints)
        #prob.solve()
        𝝰 = solve_qp(𝝨_i + 𝝨_ps, - y_i, G =-np.eye(p),h=np.zeros(p), solver="cvxopt") #x.value  # optimal dual variable
        # compute the primal (the best response) with KKT
        𝝺_i = 𝝻[i] + (1. / w[i]) * (𝝨 @ M_u) @ 𝝰
        𝝺_ps = [𝝻[perm_ps[j]] - (𝝰[j] / w[perm_ps[j]]) * 𝝨 @ e[u[j]] for j in range(p)]
        prob = (0.5) * 𝝰.T @ (𝝨_i + 𝝨_ps) @ 𝝰 - 𝝰.T @ y_i
        #return -prob.value, np.vstack([𝝺_i, 𝝺_ps]), (i, u)
        return -prob , np.vstack([𝝺_i, 𝝺_ps]), (i, u)

    return min([solve_dual_non_ps(i, d_vec) for i in non_ps for d_vec in d_vecs], key=lambda x: x[0])


def cpt_inf_ps(𝝻, 𝝨, ps, w):
    r'''
    Compute the best response and $T_1^\star(\theta, w)^{-1}$ at a given $w$ when
    we remove an arm from the ps
    :param 𝝻: arms_means
    :param 𝝨: covariance matrix
    :param ps: pareto set
    :param w:  allocation vector
    :return: T_1^\star(\theta, w)^{-1}, the best response, (arm $i$ to remove, arm $j$ that dominate $i$)
    '''
    # solve the dual problem using qpsolvers
    d = np.shape(𝝨)[0]
    def solve_dual_ps(i, j):
        r'''Solve the dual problem in $\bR^d$'''
        # i, j are two arms in the ps
        #x = cp.Variable(𝝻.shape[1])
        #constraints = [x >= 0]
        #objective = cp.Minimize((1. / 2) * (1. / w[i] + 1. / w[j]) * cp.quad_form(x, 𝝨, True) - x.T @ (𝝻[i] - 𝝻[j]))
        #prob = cp.Problem(objective, constraints)
        #prob.solve()
        𝝰 =  solve_qp((1. / w[i] + 1. / w[j]) *𝝨 , -  (𝝻[i] - 𝝻[j]), G =-np.eye(d),h=np.zeros(d), solver="cvxopt") #x.value  # optimal dual variable
        # compute the primal (the best response) with KKT
        𝝺_i = 𝝻[i] - (1. / w[i]) * (𝝨 @ 𝝰)
        𝝺_j = 𝝻[j] + (1. / w[j]) * (𝝨 @ 𝝰)
        prob = (1. / 2) * (1. / w[i] + 1. / w[j]) * 𝝰.T @ 𝝨 @𝝰 - 𝝰.T@(𝝻[i] - 𝝻[j])
        return -prob, [𝝺_i, 𝝺_j], (i, j)

    return min([solve_dual_ps(i, j) for i in ps for j in ps if i != j], key=lambda x: x[0])


def cpt_br(𝝻, 𝝨, w, ps, non_ps, d_vecs):
    r''' Compute $T^*(\theta, w)^{-1}$ and the best response at a given w
    :param 𝝻: arms_means
    :param 𝝨: covariance matrix
    :param w: allocation vector
    :param ps: pareto set
    :param non_ps: suboptimal set
    :param d_vecs: [d]^p
    :return: $T^*(\theta, w)^{-1}$ and the best response at w
    '''
    # compute the best response for a given w
    t_ps = cpt_inf_ps(𝝻, 𝝨, ps, w) if len(ps) > 1 else (np.inf, None)
    t_nps = cpt_inf_non_ps(𝝻, 𝝨, ps, non_ps, d_vecs, w) if len(non_ps) > 0 else (np.inf, None)
    inv_T_s = None
    𝝺 = np.copy(𝝻)
    if t_ps[0] < t_nps[0]:
        # best response is obtained by removing an optimal arm from the ps
        (i, j) = t_ps[-1]
        𝝺[i] = t_ps[-2][0]
        𝝺[j] = t_ps[-2][1]
        inv_T_s = t_ps[0]
    else:
        # best response is obtained by adding a suboptimal arm to the ps
        i = t_nps[-1][0]  # arm to add to the ps
        𝝺[i] = t_nps[-2][0]
        for (k, v) in enumerate(ps):
            𝝺[v] = t_nps[-2][k + 1]
        inv_T_s = t_nps[0]
    return inv_T_s, 𝝺


def cpt_lbd_correl(𝝻: np.ndarray,
                   𝝨: np.ndarray,
                   *,
                   niter=100,
                   ps_mask=None,
                   w_init=None,
                   eps=1e-9,
                   return_br=False):
    r'''
    Compute $T^*(\theta, w)^{-1}, w^\star$ and the best response at $w^\star$
    :param 𝝻: arms_means
    :param 𝝨: covariance matrix
    :param niter: number of iterations
    :param ps_mask: mask of pareto set
    :param w_init: initial allocation vector
    :param eps: minimal weights to avoid division by zero
    :param return_br: whether to return the best response
    :return: (w^*, T^*) or (w^*, T^*, the best response)
    '''
    # compute the lower bound with possible correlations
    ps_mask = is_non_dominated(𝝻) if ps_mask is None else ps_mask
    K, p, d = len(ps_mask), int(sum(ps_mask)), 𝝻.shape[-1]
    w = w_init if w_init is not None else np.random.uniform(size=K)
    w /= w.sum()
    I = np.eye(K)
    ps, non_ps = np.where(ps_mask)[0], np.where(~ps_mask)[0]
    𝝨_inv = np.linalg.inv(𝝨)
    d_vecs = list(product(range(d), repeat=p))
    # compute the lower bound using FW  with super gradients
    T_inv = None
    for k in range(niter):
        # compute the best response at and T^*(w)^{-1}
        T_inv, br = cpt_br(𝝻, 𝝨, (w + eps) / (w + eps).sum(), ps, non_ps, d_vecs)
        gd = np.array([np.vdot(br[i] - 𝝻[i], 𝝨_inv @ (br[i] - 𝝻[i])) / 2. for i in range(K)])
        s_k = I[np.argmin(-gd)]
        alpha = 2. / (k + 2)
        w = w + alpha * (s_k - w)
    return (w, 1 / T_inv, cpt_br(𝝻, 𝝨, w, ps, non_ps, d_vecs)[-1]) if return_br else (w, 1 / T_inv)


# compute  the lower bound for independent components
# using the algorithm of crepon et al 2024
## Using gradient norm as stopping cond steems instable
def cpt_lb_ind(𝝻: np.ndarray,
               *,
               niter=100,
               eps=1e-7,
               w_init=None):
    r'''
    Compute T^* for instances with unit variance and diagonal covariance using FW with supergradients
    provided by the algorithm of crepon et al 2024
    :param 𝝻: arms means
    :param niter: number of iterations
    :param eps: minimal weights to avoid division by zero
    :param w_init: initial allocation vector
    :return: $w^*, T^*$
    '''
    K, d = 𝝻.shape
    w = w_init if w_init is not None else np.random.uniform(size=K)
    w /= w.sum()
    I = np.eye(K)
    grad = PCnd(𝝻) if d > 2 else PC2d(𝝻)
    for k in range(niter):
        s_k = I[np.argmin(-grad.get_cost(w + eps / np.sum(w + eps))[1])]
        alpha = 2. / (k + 2.)
        w = w + alpha * (s_k - w)
    return w, 1. / grad.get_cost(w + eps)[0]


if __name__ == "__main__" and len(sys.argv) < 1:
    """𝝻 = arms_means = np.array([[0.2528313, 0.50961513],
                               [0.49083348, 0.04864432],
                               [0.40864574, 0.8624948],
                               [0.08210546, 0.37149325],
                               [0.99130959, 0.39292875]])
    𝝨 = cov = np.array([1., 1.])
    niter = 250
    w_star, T_star = fw(𝝻, 𝝨, None, niter, 1e-7)
    print(f"w^* = {w_star}\nT^* = {T_star}")"""
'''
parser = argparse.ArgumentParser()
parser.add_argument('-mean', type=eval)
parser.add_argument('-var', type=eval)
parser.add_argument('--wi', type=eval, default=None)
parser.add_argument('--niter', type=eval, default=100)
parser.add_argument('--eps', type=eval, default=1e-7)
args = parser.parse_args()
w_star, T_star = fw(args.mean, args.var, args.wi, args.niter, args.eps)
print(list(w_star), T_star)
'''
