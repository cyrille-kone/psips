#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "utils.hpp"
#include "bandits.hpp"
#include "policies.hpp"



policy::policy(bandit &bandit_ref): K(bandit_ref.K), d(bandit_ref.d), action_space(bandit_ref.action_space), cov(bandit_ref.cov),chol_cov(bandit_ref.chol_cov),bandit_ref(bandit_ref){
}
std::pair<bool, size_t> policy::loop(const size_t&, const double&) {
    return {};
}


peps_psi::peps_psi(bandit &bandit_ref, learner& l_ref) : policy(bandit_ref),
inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))),
l_ptr(&l_ref), A(xt::eye<dt>(bandit_ref.K)), w_exp(xt::ones<dt>({bandit_ref.K})/ ((dt) bandit_ref.K)),
h(bandit_ref.K){
    name = "PSITS";
    set_A_tilde_outer;
};

peps_psi::peps_psi(bandit& bandit_ref, learner& l_ref , const vv<dt,2>& A_): policy(bandit_ref), inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))),
l_ptr(&l_ref),A(A_), h(A_.shape()[1]), w_exp(xt::ones<dt>({bandit_ref.K})/ ((dt) bandit_ref.K)){
    name = "PSITS";
    set_A_tilde_outer;
}

peps_psi::peps_psi(bandit & bandit_ref, learner & l_ref, const vv<dt,2> &A_, const vv<dt> &w_exp_):policy(bandit_ref), inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))),
                                                                                     l_ptr(&l_ref),A(A_), h(A_.shape()[1]), w_exp(w_exp_/ xt::sum(w_exp_)(0u)){
    name = "PSITS";
    set_A_tilde_outer;
}



std::pair<bool, size_t> peps_psi::loop(const size_t& seed, const dt& delta) {
    // Thread-level initialization of the PRNG
    bandit_ref.reset_env(seed);
    // Initialize the Hedge learner
    l_ptr->init();
    // Initialization of data containers
    size_t dh = d*h;
    vv<> w_t;
    vv<> w_t_tilde;
    vv<bool> ps_mask;
    vv<dt,2> cinv_V_t_tilde; // full covariance matrix;
    vv<dt,1> vec_lbd_t,  vec_lbd_t_tilde;
    vv<dt,1> vec_theta_t; // vectorized version of theta
    size_t I_t, m, mp, i, t{1};
    dt eta_t_val{1/2.};
    dt alpha_t, alpha{1.}, M_t_val, c_t_val;
    vv<> loss_t {xt::empty<dt>({K})};
    vv<> cum_w {xt::zeros<dt>({K})};
    vv<> c_w {xt::zeros<dt>({K})}; // to remove
    vv<size_t> N_t{xt::ones<size_t>({K})}; // Pulling each arm once
    vv<dt,2> V_t {xt::eye<dt>(h)};
    vv<dt,2> V_t_inv {xt::eye<dt>(h)};
    vv<dt,1> a_t; // feature selected at time $t$;
    vv<dt,1> a_t_tilde; // modified feature selected at time $t$;
    vv<dt,1> X_t ; // current observation at time $t$;
    // Initialization
    vv<dt,2> theta_t{xt::zeros<dt>({h, d})}, lbd_t_tilde, lbd_t, pi_t;  // parameter in matrix form; d x h (d x K)
    vec_theta_t = xt::ravel<xt::layout_type::column_major>(theta_t);
    vv<dt,2> s_t {xt::zeros<dt>({h, d})}; // cumulative sum of observations
    auto inv_chol_cov = xt::linalg::cholesky(inv_cov);
    //
    while(true){
        cinv_V_t_tilde = xt::linalg::kron(chol_cov, xt::linalg::cholesky(V_t_inv));
        m = 0u;
        mp = 0ul;
        c_t_val = utils::peps::c(t, delta);
        M_t_val = utils::peps::M(t, delta);
        ps_mask = fpsi( xt::linalg::dot(A, theta_t ));
        do {
            // centered rv from $Pi_t$
            pi_t = xt::reshape_view<xt::layout_type::column_major>(xt::linalg::dot(cinv_V_t_tilde, xt::random::randn<dt>({dh}, 0., 1., gen)), {h, d});
            lbd_t = sqrt(1./eta_t_val)*pi_t + theta_t;
            lbd_t_tilde = sqrt(c_t_val)*pi_t + theta_t;
            mp += (mp != m || utils::peps::struct_in_alt(xt::transpose(lbd_t_tilde), A, ps_mask))?0ul:1ul;
            ++m;
        } while (( mp!=m ||((dt) mp < M_t_val) )&& ! utils::peps::struct_in_alt(xt::transpose(lbd_t),A, ps_mask));

        vec_lbd_t = xt::ravel<xt::layout_type::column_major>(lbd_t);
        if ((mp==m) && (dt)mp >= M_t_val) break;
        //TH_STR // Thompson or Posterior Stopping
        w_t = l_ptr->getw();
        alpha_t = std::pow(t, -alpha);
        w_t_tilde = (1.- alpha_t)*w_t + alpha_t*w_exp;
        I_t = std::discrete_distribution(w_t_tilde.begin(), w_t_tilde.end())(gen);
        X_t = xt::row(bandit_ref.sample({I_t}), 0u);
        N_t(I_t) += 1u; // to remove
        // compute bonuses
        for(auto a: action_space){
            loss_t(a) = xt::linalg::vdot(vec_theta_t - vec_lbd_t, xt::linalg::dot(xt::view(A_tilde, a, xt::all(), xt::all()), (vec_theta_t - vec_lbd_t)));
        }
        // Update learner bonus
        l_ptr->updt(-loss_t);
        // update information matrix
        a_t = xt::row(A, I_t);
        V_t += xt::view(A_outer, I_t, xt::all(), xt::all());
        // Shermann Morrison update
        a_t_tilde = xt::linalg::dot(V_t_inv, a_t);
        V_t_inv -= xt::linalg::outer(a_t_tilde, a_t_tilde) / (1. + xt::linalg::vdot(a_t_tilde, a_t));
        s_t += xt::linalg::outer(a_t, X_t);
        theta_t = xt::linalg::dot(V_t_inv, s_t);
        vec_theta_t = xt::ravel<xt::layout_type::column_major>(theta_t);
        ++t;
    }
    return return_cr_sc(ps_mask, bandit_ref.ps_mask, N_t);
}




ape::ape(bandit &bandit_ref, const vv<dt,2> &A_): policy(bandit_ref), A(A_), h(A_.shape()[1ul]), inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))){
    name="APE";
set_A_tilde_outer;
};
ape::ape(bandit &bandit_ref): policy(bandit_ref), h(bandit_ref.K),
A(xt::eye<dt>(bandit_ref.K)),inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))){
    name = "APE";
    set_A_tilde_outer;
};
std::pair<bool, size_t> ape::loop(const size_t &seed, const dt& delta) {
    // Thread-level initialization of the PRNG
    bandit_ref.reset_env(seed);
    // Initialize data containers
    dt M_t;
    vv<dt,2> V_t {xt::eye<dt>(h)};
    vv<dt,2> V_t_inv {xt::eye<dt>(h)};
    vv<dt,1> a_t; // feature selected at time $t$;
    vv<bool> ps_mask;
    vv<dt,2> cinv_V_t_tilde; // full covariance matrix;
    vv<dt,1> a_t_tilde; // modified feature selected at time $t$;
    vv<dt,2> theta_t{xt::zeros<dt>({h, d})}, lbd_t_tilde,  pi_t;  // parameter in matrix form; d x h (d x K)
    vv<dt,2> s_t {xt::zeros<dt>({h, d})}; // cumulative sum of observations
    //
    vv<dt> n_vec =  xt::sqrt(xt::diagonal(cov)); // normalize the observations for confidence intervals
    size_t bt, ct, I_t, m, t{1};
    vv<size_t> N_t{xt::zeros<size_t>({K})};
    vv<dt> beta_vec{xt::zeros<dt>({K}) + INF};
    vv<dt,2> mu_t{xt::zeros<dt>({K, d})};
    while(true){
        //TH_STR
        ps_mask = fpsi(mu_t);
        if (utils::ape::cpt_z1_z2(mu_t, ps_mask, beta_vec)>=0.) break;
        bt = utils::ape::get_bt(mu_t, ps_mask, beta_vec);
        ct = utils::ape::get_ct(mu_t, bt, beta_vec);
        // we can sample both bt and ct to be faster
        I_t = (N_t(bt) < N_t(ct))? bt: ct;
        s_t += xt::linalg::outer(xt::row(A, I_t), xt::row(bandit_ref.sample({I_t}), 0ul)/n_vec);
        N_t(I_t) += 1;
        beta_vec(I_t) = utils::ape::beta(N_t(I_t), delta);

        a_t = xt::row(A, I_t);
        V_t += xt::view(A_outer, I_t, xt::all(), xt::all());
        a_t_tilde = xt::linalg::dot(V_t_inv, a_t);
        V_t_inv -= xt::linalg::outer(a_t_tilde, a_t_tilde) / (1. + xt::linalg::vdot(a_t_tilde, a_t));
        theta_t = xt::linalg::dot(V_t_inv, s_t);
        mu_t = xt::linalg::dot(A, theta_t );
        ++t;
    }
   return return_cr_sc(ps_mask, bandit_ref.ps_mask, N_t);
}

// Generate functions for batch run

FACTORY_BATCH(ape)
FACTORY_BATCH(oracle)
FACTORY_BATCH(peps_psi)
FACTORY_BATCH(rr)

