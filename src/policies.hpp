#pragma once
#include "utils.hpp"
#include "bandits.hpp"
#include "learners.hpp"
#include <omp.h>

# define set_A_tilde_outer \
A_tilde = xt::empty<dt>({K, d*h, d*h}); \
A_outer = xt::empty<dt>({K, h, h}); \
for (size_t i=0; i<K; ++i){             \
xt::view(A_outer, i, xt::all(), xt::all()) = xt::linalg::outer(xt::row(A, i), xt::row(A, i));                     \
xt::view(A_tilde, i, xt::all(), xt::all()) = xt::linalg::kron(inv_cov, xt::view(A_outer, i, xt::all(), xt::all()));       \
}

#define TH_STR {M_t = utils::peps::M(t, delta); m = 0ul; c_t_val = utils::peps::c(t, delta); lambda_t = gen_instance(mu_t, chol_cov, Ns); \
     ps_mask = fpsi(mu_t); lambda_t = sqrt(c_t_val)*lambda_t + (1.-sqrt(c_t_val))*mu_t ;\
while((dt)m < M_t && !utils::peps::in_alt(lambda_t, ps_mask)){ m+=1; lambda_t = utils::peps::gen_i(mu_t, chol_cov, Ns);} if (dt(m)>=M_t or t>500'000) break;}


#define FACTORY_BATCH(fn)  vv<size_t,2> batch_##fn (const fn& pol, const bandit& bandit_ref, dt delta, const vv<size_t>& seeds) \
{ auto ans{xt::empty<size_t>({seeds.size(), 2ul})}; \
  fn psi{pol};                              \
  size_t i;                                           \
  std::pair<bool, size_t> res;                          \
  _Pragma("omp parallel for private(i) default(none) shared(seeds, ans, delta)  firstprivate(psi, res) num_threads(4)") \
  for(i=0;i<seeds.size(); ++i){                          \
  res = psi.loop(seeds[i], delta);                        \
  ans(i, 0) = res.first;                                                                        \
  ans(i, 1) = res.second;}                                 \
  return ans; }                  \

struct policy{
    const size_t& K; // number of arms
    const size_t& d; // number of objectives
    const vv<dt,2>& cov; // covariance matrix
    const vv<dt,2>& chol_cov; //cholesky of the covariance matrix
    const vv<size_t>& action_space;
    std::string name;
    bandit& bandit_ref;
    //policy() = default;
    explicit policy(bandit&);
    virtual std::pair<bool,size_t> loop(const size_t& seed, const dt& delta)=0;
};


struct peps_psi: policy{
    const vv<dt,2> inv_cov; // this object is not declared in bandit struct
    const vv<dt,2> A; // features matrix
    vv<dt,3> A_tilde; //
    vv<dt,3> A_outer; //
    const vv<dt> w_exp ; // forced exploration vector
    const size_t h; // dimension of features
    learner* l_ptr;
    std::unique_ptr<learner> u_ptr;
    explicit peps_psi(bandit&) = delete;
    peps_psi(bandit&, learner&);
    peps_psi(bandit&, learner&, const vv<dt,2>&);
    peps_psi(bandit& bandit_, learner& learner_, const vv<dt,2>& A, const vv<dt>& w_exp);
    [[nodiscard]] std::pair<bool,size_t> loop(const size_t&, const dt& delta) override;
    peps_psi(const peps_psi& other): policy(other.bandit_ref),
    inv_cov(other.inv_cov), A(other.A), w_exp(other.w_exp), h(other.h), A_outer(other.A_outer), A_tilde(other.A_tilde){
        u_ptr = std::unique_ptr<learner>(other.l_ptr->cpy());
        l_ptr = u_ptr.get();
        name = "PSITS";
    }
};



struct ape: policy{
    ape() = delete;
    explicit ape(bandit&);
    ape(bandit& bandit_ref, const vv<dt,2>& A_);
    [[nodiscard]] std::pair<bool, size_t> loop(const size_t&, const dt&) override;
    const size_t h;
    const vv<dt,2> A;
    vv<dt,3> A_tilde; //
    vv<dt,3> A_outer; //
    const vv<dt,2> inv_cov; // this object is not declared in bandit struct
    // copy ctor
    ape(const ape& other): policy(other.bandit_ref), A(other.A),
    A_tilde(other.A_tilde), A_outer(other.A_outer), inv_cov(other.inv_cov), h(other.h){
        name = "APE";
    }
};

struct oracle:policy{
    const vv<dt> w_star;
    const size_t h; // dimension of features
    const vv<dt,2> inv_cov; // this object is not declared in bandit struct
    const vv<dt,2> A; // features matrix
    vv<dt,3> A_tilde; //
    vv<dt,3> A_outer; //

    oracle(bandit& bandit_ref, const vv<dt>& w_star):policy(bandit_ref), w_star(w_star),
    h(bandit_ref.K), A(xt::eye<dt>(bandit_ref.K)),inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))){
        name ="ORACLE";
        set_A_tilde_outer;
    };
    oracle(bandit& bandit_ref, const vv<dt>& w_star, const vv<dt,2>& A_):policy(bandit_ref), w_star(w_star),h(A_.shape()[1]), A(A_),inv_cov(xt::linalg::pinv(bandit_ref.cov + 1e-9* xt::eye(bandit_ref.d))){
        name ="ORACLE";
        set_A_tilde_outer;
    };
    oracle(const oracle& other): policy(other.bandit_ref), w_star(other.w_star),h(other.h),
    A(other.A), inv_cov(other.inv_cov), A_outer(other.A_outer), A_tilde(other.A_tilde){
        name ="ORACLE";
    }
    [[nodiscard]] std::pair<bool,size_t> loop(const size_t& seed, const dt& delta) override{
        bandit_ref.reset_env(seed);
        size_t dh = d*h;
        vv<size_t> N_t{xt::zeros<size_t>({K})};
        vv<bool> ps_mask;
        dt c_t_val;
        vv<dt,2> V_t {xt::eye<dt>(h)};
        vv<dt,2> V_t_inv {xt::eye<dt>(h)};
        vv<dt,1> a_t; // feature selected at time $t$;
        dt M_t_val;
        size_t I_t, t{1}, m;
        vv<dt,2> cinv_V_t_tilde; // full covariance matrix;
        vv<dt,1> a_t_tilde; // modified feature selected at time $t$;
        vv<dt,1> X_t ; // current observation at time $t$;
        vv<dt,2> theta_t{xt::zeros<dt>({h, d})}, lbd_t_tilde,  pi_t;  // parameter in matrix form; d x h (d x K)
        vv<dt,2> s_t {xt::zeros<dt>({h, d})}; // cumulative sum of observations
        while (true){
            c_t_val = utils::peps::c(t, delta);
            M_t_val = utils::peps::M(t, delta);
            cinv_V_t_tilde = xt::linalg::kron(chol_cov, xt::linalg::cholesky(V_t_inv));
            // centered rv from $Pi_t$
            ps_mask = pareto_optimal_arms_mask( xt::linalg::dot(A, theta_t ));
            m=0ul;
            do {
                pi_t = xt::reshape_view<xt::layout_type::column_major>(xt::linalg::dot(cinv_V_t_tilde, xt::random::randn<dt>({dh}, 0., 1., gen)), {h, d});
                lbd_t_tilde = sqrt(c_t_val)*pi_t + theta_t;
                ++m;
            }while((((dt) m < M_t_val) )&& ! utils::peps::struct_in_alt(xt::transpose(lbd_t_tilde),A, ps_mask));
            if ((dt)m >= M_t_val) break;//*/
            //TH_STR
            I_t = std::discrete_distribution(w_star.begin(), w_star.end())(gen);
            X_t = xt::row(bandit_ref.sample({I_t}), 0u);
            N_t(I_t) += 1u;
            // update means and information matrix
            a_t = xt::row(A, I_t);
            V_t += xt::view(A_outer, I_t, xt::all(), xt::all());
            // Shermann Morrison update
            a_t_tilde = xt::linalg::dot(V_t_inv, a_t);
            V_t_inv -= xt::linalg::outer(a_t_tilde, a_t_tilde) / (1. + xt::linalg::vdot(a_t_tilde, a_t));
            s_t += xt::linalg::outer(a_t, X_t);
            theta_t = xt::linalg::dot(V_t_inv, s_t);
            ++t;
        }
        return return_cr_sc(ps_mask, bandit_ref.ps_mask, N_t);
    }
};
struct rr: oracle{
    explicit rr(bandit& bandit_ref): oracle(bandit_ref, xt::ones<dt>({bandit_ref.K})/(dt)bandit_ref.K){
        name="RR";
    };
    rr(bandit& bandit_ref, const vv<dt,2>& A_): oracle(bandit_ref, xt::ones<dt>({bandit_ref.K})/(dt)bandit_ref.K , A_){
        name="RR";
    }
    rr(const rr& other): oracle(other.bandit_ref, other.w_star, other.A){
        name ="RR";
    }


};

vv<size_t,2> batch_peps_psi(const peps_psi& pol, const bandit& bandit_ref, dt delta, const vv<size_t>& seeds);
vv<size_t,2> batch_ape(const ape& pol, const bandit& bandit_ref, dt delta, const vv<size_t>& seeds);
vv<size_t,2> batch_rr(const rr& pol, const bandit& bandit_ref, dt delta, const vv<size_t>& seeds);
vv<size_t,2> batch_oracle(const oracle& pol, const bandit& bandit_ref, dt delta, const vv<size_t>& seeds);
