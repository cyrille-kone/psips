#include "fcts.hpp"

/*
 * functional implementation of the peps-psi policy
 * measures multiple quantities of interest
 * preferred this for unstructured instances
 */
vv<size_t,2ul> fun_peps(size_t seed,
                  bandit& b_ref,   // bandit instance
                  learner& l_ref, // max learner
                  size_t t_min, // forced  minimum number of iterations
                  size_t t_max,// maximum number of iterations
                  dt delta,    // confidence level
                  size_t m_alloc, // minimum pre-allocation for the vector of returns
                  dt alpha // value of $\alpha$ for forced exploration
){
    b_ref.reset_env(seed); //seeding the PRNG
    l_ref.init(); // resetting the max_learner
    // initialization of data containers
    auto inv_cov = xt::linalg::inv(b_ref.cov);
    auto& K = b_ref.K;
    auto& chol_cov = b_ref.chol_cov;
    vv<dt,2> lambda_t;
    auto total_outcome = b_ref.sample(b_ref.action_space);
    vv<dt,2> mu_t{total_outcome/ 1.};
    vv<size_t> N_t{xt::ones<size_t>({K})};
    vv<bool> ps_mask;
    vv<dt> w_t{xt::ones<dt>({K})/(dt)K};
    vv<> loss_t{xt::empty<dt>({K})};
    vv<> w_exp{xt::ones<dt>({K})/(dt)K};
    vv<> w_t_tilde;
    size_t I_t, m, mp, i, t{K}, tt{0ul}; // tt is the un-initialized counter of rounds
    dt alpha_t_val,  M_t_val, c_t_val, eta_t_val{1./1.}; // prescribed by theory
    vv<> cum_w = {xt::zeros_like(loss_t)};
    bool c_stop{false}; // whether to check stopping
    vv<size_t,2> r_tens;// result tensor
    vv<size_t,2> a_tens; // allocator tensor
    // when t_min = t_max create a tensor for the results
    // allocate memory for the minimum duration
    if (t_min > K || m_alloc>0ul){
        // check also that t_min is finite
        // allocate space in the tensor of results
        r_tens = xt::empty<dt>({(size_t)std::max(((int)t_min-(int)K),(int) m_alloc), 6ul});
    }
    while(t<t_max){
        // measure the whole time of the algorithm loop
        auto s = std::chrono::steady_clock::now();
        // update Pareto set
        ps_mask = fpsi(mu_t);
        c_stop = c_stop || (t>=t_min); // whether to activate the stopping rule
        M_t_val = utils::peps::M(t, delta);
        m= 0ul;
        mp = 0ul;
        c_t_val = utils::peps::c(t, delta);
        // measure the time needed for  finding a parameter in the alternative
        auto ss = std::chrono::steady_clock::now();
        // compute alternative parameter
        do{
            lambda_t = utils::peps::gen_i(mu_t, chol_cov, N_t);
            mp += (mp != m || utils::peps::in_alt(sqrt(c_t_val)*lambda_t + (1.-sqrt(c_t_val))*mu_t, ps_mask))?0ul:1ul;
            ++m;
        }
        while( (!c_stop || ( mp!=m ||((dt) mp < M_t_val) )) && !utils::peps::in_alt(sqrt(1./eta_t_val)*lambda_t + (1.-sqrt(1./eta_t_val))*mu_t, ps_mask));
        // end of the second clock; measure of the time spent finding an alternative
        auto ee = std::chrono::steady_clock::now();
        /*
         * Posterior based sampling rule
         * Tracking or sampling
         */
        // sample from w_t
        alpha_t_val = std::pow(t, -alpha);
        w_t_tilde = (1.- alpha_t_val)*w_t + alpha_t_val*w_exp;
        // sampling
        // I_t = std::discrete_distribution(w_t_tilde.begin(), w_t_tilde.end())(gen);
        // tracking
        cum_w +=  w_t_tilde;
        I_t = xt::argmin(N_t - cum_w)(0ul);
        // update the vector of means
        upt_means(mu_t, total_outcome, I_t, N_t, b_ref)
        i = 0ul;
        // [maybe] replace with simple for loop
        // compute losses
        std::generate(loss_t.begin(), loss_t.end(), [&i, &mu_t, &lambda_t,&inv_cov]()mutable {
            return xt::linalg::vdot(xt::row(mu_t , i) - xt::row(lambda_t, i), xt::linalg::dot(inv_cov, xt::row(mu_t, i)- xt::row(lambda_t, ++i)));
        });
        // max_learner incurs bonuses
        l_ref.updt(-loss_t);
        // end of second clock; measuring time spent on the whole iteration
        auto e = std::chrono::steady_clock::now();
        // compute the durations
        r_tens(tt, 0ul) =  m;
        r_tens(tt, 1ul) =  mp;
        r_tens(tt, 2ul) = iterator_eql(ps_mask, b_ref.ps_mask);
        r_tens(tt, 3ul) = cpt_duration(s, e, std::chrono::nanoseconds);
        r_tens(tt, 4ul) = cpt_duration(ss, ee, std::chrono::nanoseconds);
        r_tens(tt, 5ul) =  I_t;
        // check if we can stop when the stopping is activated

        if (c_stop && ((mp==m) && (dt)mp >= M_t_val)) break;
        // internal time counters
        ++t;
        ++tt;
        // check if new memory allocation is needed, allocate and concatenate
        if (tt == r_tens.shape()[0] && t<t_max){
            // allocate new memory and concatenate
            a_tens = xt::empty_like(r_tens);
            // concatenate with previous tensor
            r_tens = xt::concatenate(xt::xtuple(r_tens, a_tens));
        }
    }
    // return check if stopping is due to t_max nope, can be deduced from data
    return xt::view(r_tens, xt::range(_,tt), xt::all());
}


/*
 * functional implementation of APE
 */
vv<size_t,2ul> fun_ape(size_t seed,
                 bandit& b_ref,   // bandit instance
                 size_t t_min, // forced  minimum number of iterations
                 size_t t_max,// maximum number of iterations
                 dt delta,    // confidence level
                 size_t m_alloc // minimum pre-allocation for the vector of returns
){
    // Thread-level initialization of the PRNG
    b_ref.reset_env(seed);
    // Initialize data containers
    dt M_t_val;
    auto& K = b_ref.K;
    vv<bool> ps_mask;
    //vv<dt,2ul> lambda_t; //
    size_t I_t, m, t{K}, tt{0ul}; // tt is the un-initialized counter of rounds
    size_t b_t, c_t;
    vv<size_t> N_t{xt::ones<size_t>({K})};
    vv<dt> beta_vec{xt::zeros<dt>({K}) + utils::ape::beta(1, delta)};
    // Initializing means by sampling each arm once
    vv<dt,2> total_outcome {b_ref.sample(b_ref.action_space)};
    vv<dt,2> mu_t{total_outcome / 1.};
    bool c_stop{false}; // whether to check stopping
    bool t_stop; // temporary variable to hold stopping
    vv<size_t,2> r_tens;// result tensor
    vv<size_t,2> a_tens; // allocator tensor
    // when t_min = t_max create a tensor for the results
    // allocate memory for the minimum duration
    if (t_min > K || m_alloc>0ul){
        // check also that t_min is finite
        // allocate space in the tensor of results
        r_tens = xt::empty<dt>({(size_t)std::max(((int)t_min-(int)K),(int) m_alloc), 6ul});
    }
    // measure the whole time of the algorithm loop
    while(t<t_max){
        auto s = std::chrono::steady_clock::now();
        ps_mask = fpsi(mu_t);
        c_stop = c_stop || (t>=t_min); // whether to activate the stopping rule
        t_stop = c_stop && utils::ape::cpt_z1_z2(mu_t, ps_mask, beta_vec)>=0.;
        // start of the second clock to measure sampling rule duration
        auto ss = std::chrono::steady_clock::now();
        b_t = utils::ape::get_bt(mu_t, ps_mask, beta_vec);
        c_t = utils::ape::get_ct(mu_t, b_t, beta_vec);
        // we sample both bt and ct to be faster
        I_t = (N_t(b_t) < N_t(c_t))? b_t: c_t;
        // end of the second clock
        auto ee = std::chrono::steady_clock::now();
        for (auto a: {I_t}) {
            upt_means(mu_t,total_outcome,a,N_t, b_ref);
            beta_vec(a) = utils::ape::beta(N_t(a), delta);
        }
        // end of the first clock
        auto e = std::chrono::steady_clock::now();
        // compute the durations
        r_tens(tt, 0ul) = 0ul; // not define if thompson stopping is not used
        r_tens(tt, 1ul) = 0ul; // not define if thompson stopping is not used
        r_tens(tt, 2ul) = iterator_eql(ps_mask, b_ref.ps_mask);
        r_tens(tt, 3ul) = cpt_duration(s, e, std::chrono::nanoseconds);
        r_tens(tt, 4ul) = cpt_duration(ss, ee, std::chrono::nanoseconds);
        r_tens(tt, 5ul) =  I_t;
        // check if we can stop when the stopping is activated
        if (t_stop) break;
        // internal time counters
        ++t;
        ++tt;
        // check if new memory allocation is needed, allocate and concatenate
        if (tt == r_tens.shape()[0] && t<t_max){
            // allocate new memory and concatenate
            a_tens = xt::empty_like(r_tens);
            // concatenate with previous tensor
            r_tens = xt::concatenate(xt::xtuple(r_tens, a_tens));
        }
    }
    return xt::view(r_tens, xt::range(_,tt), xt::all());
}
