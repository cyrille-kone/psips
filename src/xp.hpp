#pragma once
#include "utils.hpp"
#include "fcts.hpp"
#include "noc.hpp"

#define DEFINE_INSTANCE_K_D(name, K, d) \
inline const size_t name##_K = K;\
inline const size_t name##_d = d; \
/*
 * COVBOOST
 * @ref  Munro et al
 */
inline const size_t COV_BOOST_K = 20;
inline const size_t COV_BOOST_d = 3;
inline const vv<dt,2> COV_BOOST_MEANS = {{ 9.50479943,  6.85646198,  4.56226268},
                                         { 9.29302574,  6.64118217,  4.03600899},
                                         { 9.05368656,  6.40687999,  3.56388296},
                                         {10.21251518,  7.48941208,  4.42843301},
                                         {10.04680837,  7.19967835,  4.36182393},
                                         { 8.34379173,  5.66642669,  3.51154544},
                                         { 8.22174773,  5.45532112,  3.64021428},
                                         { 9.74560492,  7.2730926 ,  4.7095302 },
                                         {10.42726889,  7.61035762,  4.71849887},
                                         { 8.93761259,  6.18826412,  3.84374416},
                                         { 7.80669637,  5.26269019,  3.97029191},
                                         { 8.85008761,  6.58892648,  4.7335634 },
                                         { 8.4411757 ,  6.15273269,  4.5890408 },
                                         { 9.92900909,  7.39079852,  4.74927053},
                                         { 9.68315255,  7.20340552,  4.91191932},
                                         { 7.51479976,  5.3082677 ,  3.95508249},
                                         { 7.26542972,  4.99043259,  4.01638302},
                                         { 8.61558951,  6.33327963,  4.66343909},
                                         {10.34531673,  7.769801  ,  5.00327494},
                                         { 8.29304914,  5.92157842,  3.86702564}};

inline const vv<dt,2> COV_BOOST_COV = {{0.70437039, 0., 0.},
                                       {0., 0.82845749, 0.},
                                       {0., 0., 1.53743137}};

inline const vv<dt,1> COV_BOOST_W_STAR = {0.00768655, 0.0015972 , 0.00069878, 0.02295982, 0.00479161,
                                          0.00065885, 0.00078862, 0.01796856, 0.13975543, 0.00109808,
                                          0.00139755, 0.02096331, 0.00888445, 0.02495633, 0.34938857,
                                          0.00139755, 0.00149738, 0.01297729, 0.37933616, 0.0011979};

/*
 * Instance on correlated setting used in the main paper
 */
DEFINE_INSTANCE_K_D(CORRELATION, 5, 2)

inline const vv<dt,2> CORRELATION_MEANS = {{ 0.72875559,  1.20119222},
{ 0.45524805, -0.63317069},
{ 0.62826926,  1.27683777},
{ 0.94570734,  2.31592981},
{ 2.08131887,  1.4809387 }};
inline const vv<dt,2> CORRELATION_COV = {{1., 0}, // add correlation coeff in algo
                                        {0., 1.}};
inline const vv<dt,1> CORRELATION_W_STAR {};
/*
 * Bernoulli instance defined in the main paper
 */
DEFINE_INSTANCE_K_D(BERNOULLI, 5, 2)
inline const vv<dt,2> BERNOULLI_MEANS = {{0.2528313 , 0.50961513},
                                         {0.49083348, 0.04864432},
                                         {0.40864574, 0.8624948 },
                                         {0.08210546, 0.37149325},
                                         {0.99130959, 0.39292875}};

inline const vv<dt,2> BERNOULLI_COV = {{1., 0},
                                       {0., 1.}};
inline const vv<dt,1> BERNOULLI_W_STAR {};

/*
 * EXAMPLE FOR TESTING
 */
DEFINE_INSTANCE_K_D(TEST, 5, 2)
inline const vv<dt,2> TEST_MEANS =  {{0.2528313 , 0.50961513},
                                     {0.49083348, 0.04864432},
                                     {0.40864574, 0.8624948 },
                                     {0.08210546, 0.37149325},
                                     {0.99130959, 0.39292875}};
inline const vv<dt,2> TEST_COV = {{0.5, 0},
                                  {0., .5}};

inline const vv<dt,1> TEST_W_STAR {};


namespace xp::main::bernoulli{
    /*
     * run experiment on a bernoulli instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness of each algo]
     */
    static dt delta = 0.01f; // confidence level
    static size_t niter = 500ul; // number of iterations
    static std::string out_rep = "../out/main/bern/";
    static std::string f_name = "mbern.json";
    vv<size_t,2> res_peps, res_ape, res_rr;
    size_t d_peps, d_ape, d_rr; // time elapsed on each algo
    inline int run(){
        // define bandit instance
        struct bernoulli bern(BERNOULLI_MEANS, BERNOULLI_COV);
        // define the algorithms
        ada_hedge adh{BERNOULLI_K};
        ape pape {bern};
        peps_psi ppsi{bern, adh};
        rr prr{bern};
        vv<size_t> seeds = xt::arange<size_t>(niter);
        // batch run each algo
        duration_of_instruction(batch_peps_psi(ppsi, bern,delta, seeds), res_peps, d_peps);
        duration_of_instruction(batch_ape(pape, bern, delta, seeds), res_ape, d_ape);
        duration_of_instruction(batch_rr(prr, bern, delta, seeds), res_rr, d_rr);
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps
        jsn[ppsi.name]["result"] = res_peps;
        jsn[ppsi.name]["duration"] = d_peps;
        // ape
        jsn[pape.name]["result"] = res_ape;
        jsn[pape.name]["duration"] = d_ape;
        //rr
        jsn[prr.name]["result"] = res_rr;
        jsn[prr.name]["duration"] = d_rr;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
        /*
        std::cout<<"executed perfectly with delta= "<<delta<<"\n";
        std::cout<< "APE: "<<xt::mean(res_ape, {0})<<"\n";
        std::cout<<"PEPS: "<< xt::mean(res_peps, {0})<<"\n";*/
    }
}
namespace xp::main::chrono::empM{
    /*
     * run experiment on a given instance to measure the number of rejection samples per round
     * @parameters [delta, Sigma]
     * @Algos [PEPS]
     * @Max learner[AdaHedge]
     * @Save [time, correctness and M(for peps)]
     * RUN IN SEQUENTIAL TO AVOID BORDER EFFECTS
     */
    static dt delta = 0.1f; // confidence level
    static size_t niter = 1000ul; // number of iterations
    static size_t T = 1500ul;
    static std::string out_rep = "../out/main/empM/";
    static std::string f_name = "mempM2.json";
    static const dt ANGLE  = M_PI/5.;
    static const size_t K  = 5;
    static const size_t d = 2;
    static const vv<dt,2> P{{cos(ANGLE), -sin(ANGLE)}, {sin(ANGLE), cos(ANGLE)}};
    vv<dt,2> means = xt::empty<dt>({K, d});
    static const vv<dt,2> cov = 0.5*xt::eye<dt>(d);
    inline int run() {
        xt::row(means, 0ul) = vv<dt>{1., 1.};
        for (size_t i=1ul; i<K; ++i){
            xt::row(means, i) = (vv<dt>)xt::linalg::dot(P, xt::row(means, i-1));};
        gaussian gauss(means, cov);
        ada_hedge adh{K};
        peps_psi ppsi{gauss, adh};
        // generate seeds
        vv<size_t> seeds = xt::arange<size_t>(niter);
        vv<dt,3> res_peps = xt::empty<dt>({niter, T-K, 6ul});
        for (size_t i = 0; i < niter; ++i) {
            xt::view(res_peps, i, xt::all(), xt::all()) = fun_peps(seeds(i), gauss, adh, T, T, delta, T);
        }
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps @ ape
        jsn["algo"][ppsi.name]["result"] = res_peps;
        jsn["meta"]["K"] = K;
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["arms_means"] = means;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}
// function to measure time for a given budget
namespace xp::main::covboost::chrono{
    /*
     * run experiment on a given instance to measure the timestep duration
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE]
     * @Max learner[AdaHedge]
     * @Save [time, correctness and M(for peps)]
     * RUN IN SEQUENTIAL TO AVOID BORDER EFFECTS
     */
    static dt delta = 0.1f; // confidence level
    static size_t niter = 500ul; // number of iterations
    static size_t T = 500ul;
    static std::string out_rep = "../out/main/cov_boost/";
    static std::string f_name = "mcovboost_chrono.json";

    inline int run() {
        gaussian gauss(COV_BOOST_MEANS, COV_BOOST_COV);
        ada_hedge adh{COV_BOOST_K};
        peps_psi ppsi{gauss, adh};
        ape pape{gauss};
        // generate seeds
        vv<size_t> seeds = xt::arange<size_t>(niter);
        vv<dt,3> res_ape = xt::empty<dt>({niter, T-COV_BOOST_K, 6ul});
        vv<dt,3> res_peps = xt::empty<dt>({niter, T-COV_BOOST_K, 6ul});
        for (size_t i = 0; i < niter; ++i) {
           xt::view(res_ape, i, xt::all(), xt::all()) = fun_ape(seeds(i), gauss, T, T, delta, T);
            xt::view(res_peps, i, xt::all(), xt::all()) = fun_peps(seeds(i), gauss, adh, T, T, delta, T);
        }
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps @ ape
        jsn["algo"][ppsi.name]["result"] = res_peps;
        jsn["algo"][pape.name]["result"] = res_ape;
        jsn["meta"]["K"] = COV_BOOST_K;
        jsn["meta"]["delta"] = delta;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
    // save the data to json file
}

//experiment on covboost in the main
namespace xp::main::covboost::fc{
    /*
     * run experiment on the cov_boost instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform, Oracle]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness of each algo]
     */
    static dt delta = 0.01f; // confidence level
    static size_t nruns = 50ul; // number of runs
    static std::string out_rep = "../out/main/cov_boost/";
    static std::string f_name = "mcovboost.json";
    vv<size_t,2> res_peps, res_ape, res_rr, res_orcl;
    size_t d_peps, d_ape, d_rr, d_orcl; // time elapsed on each algo
    // define bandit instance
    inline int run(){
        //define bandit instance
        gaussian gauss(COV_BOOST_MEANS, COV_BOOST_COV);
        // define the algorithms
        ada_hedge adh{COV_BOOST_K};
        ape pape {gauss};
        peps_psi ppsi{gauss, adh};
        rr prr{gauss};
        oracle orcl{gauss, COV_BOOST_W_STAR};
        vv<size_t> seeds = xt::arange<size_t>(nruns);
        // batch run each algo
        duration_of_instruction(batch_peps_psi(ppsi, gauss,delta, seeds), res_peps, d_peps);
        duration_of_instruction(batch_ape(pape, gauss, delta, seeds), res_ape, d_ape);
        duration_of_instruction(batch_rr(prr, gauss, delta, seeds), res_rr, d_rr);
        duration_of_instruction(batch_oracle(orcl, gauss, delta, seeds), res_orcl, d_orcl);
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps
        jsn["algo"][ppsi.name]["result"] = res_peps;
        jsn["algo"][ppsi.name]["duration"] = d_peps;
        // ape
        jsn["algo"][pape.name]["result"] = res_ape;
        jsn["algo"][pape.name]["duration"] = d_ape;
        //rr
        jsn["algo"][prr.name]["result"] = res_rr;
        jsn["algo"][prr.name]["duration"] = d_rr;
        // oracle
        jsn["algo"][orcl.name]["result"] = res_orcl;
        jsn["algo"][orcl.name]["duration"] = d_orcl;
        // metadata
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["nruns"] = nruns;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}


//experiment on covboost in the main
namespace xp::main::noc{
    /*
     * run experiment on the cov_boost instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness of each algo]
     */
    static dt delta = 0.1f; // confidence level
    static size_t nruns = 4ul; // number of runs
    static std::string out_rep = "../out/main/noc/";
    static std::string f_name = "mnoc.json";
    vv<size_t,2> res_peps, res_ape, res_rr;
    size_t d_peps, d_ape, d_rr; // time elapsed on each algo
    // define bandit instance
    inline int run(){
        //define bandit instance
        gaussian gauss(NOC_MEANS, NOC_COV);
        // define the algorithms
        ada_hedge adh{NOC_K};
        ape pape {gauss};
        peps_psi ppsi{gauss, adh,NOC_FEATURES, NOC_W_EXP};
        rr prr{gauss};
        vv<size_t> seeds = xt::arange<size_t>(nruns);
        // batch run each algo
        duration_of_instruction(batch_rr(prr, gauss, delta, seeds), res_rr, d_rr);
        duration_of_instruction(batch_peps_psi(ppsi, gauss,delta, seeds), res_peps, d_peps);
        duration_of_instruction(batch_ape(pape, gauss, delta, seeds), res_ape, d_ape);

        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps
        jsn["algo"][ppsi.name]["result"] = res_peps;
        jsn["algo"][ppsi.name]["duration"] = d_peps;
        // ape
        jsn["algo"][pape.name]["result"] = res_ape;
        jsn["algo"][pape.name]["duration"] = d_ape;
        //rr
        jsn["algo"][prr.name]["result"] = res_rr;
        jsn["algo"][prr.name]["duration"] = d_rr;
        // metadata
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["nruns"] = nruns;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}

// bayesian experiment in the main
namespace xp::main::bayesian::gaussian{
    /*
     * run experiment on the cov_boost instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform, Oracle]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness & size_of_ps of each algo]
     */
    static dt delta = 0.01f;
    static size_t ninst= 100; // number of instances to generate
    static size_t K = 5;
    static size_t d = 2;
    static dt min_cplx = 100.;
    static dt max_cplx = 500.;
    static std::string out_rep = "../out/main/bayes/";
    static std::string f_name = "mbayes_gaussian.json";
    static size_t seed = 42ul;
    static vv<dt,2> cov = 0.5 * xt::eye<dt>(d); // covariance is fixed and we make mu vary
    vv<size_t,3ul> r_tens;
    vv<dt,3> all_arms_means;
    vv<dt,2> arms_means;
    std::string ppsi_name, pape_name, prr_name;
    inline int run(){
        ada_hedge adh{K};
        r_tens = xt::empty<size_t>({3ul,ninst, 3ul});
        all_arms_means = xt::empty<dt>({ninst,K, d});
        size_t ctr = 0; // number of instances already executed
        dt cplx;
        size_t p;
        bool do_once = false;
        do {
            // generate parameter randomly
            // check if the complexity is inside the good range
            do {
                arms_means = xt::random::rand<dt>({K, d}, -1, 1);
                cplx = utils::psi::compute_cplxty(arms_means);
            } while(cplx>max_cplx || cplx <min_cplx);
            std::cout<<"instance found "<<cplx<<"\n";
           // define bandit and run algos
           struct gaussian gauss(arms_means, cov);
            p = gauss.ps.size();
           // define bandit algos
           // [APE, PEPS, Uniform]
           peps_psi ppsi(gauss,adh);
           ape pape(gauss);
           rr prr(gauss);
           if (!do_once){
           // save the names
           pape_name = pape.name;
           ppsi_name = ppsi.name;
           prr_name = prr.name;
           do_once = true; }
           // ape
# define exec_algo(idx, algo) \
{ auto res = (algo).loop(seed, delta); \
    xt::view(r_tens, (idx), ctr, xt::all()) = vv<size_t>{res.first, res.second, p} ;}
            exec_algo(0ul, pape);
            exec_algo(1ul, ppsi);
            exec_algo(2ul, prr);
            xt::view(all_arms_means, ctr, xt::all(), xt::all()) = arms_means;
            ++ctr;
        } while (ctr < ninst);
        // parse and save the data
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json ape
        jsn["algo"][pape_name]["result"] = xt::view(r_tens, 0ul, xt::all(), xt::all(), xt::all());
        jsn["algo"][ppsi_name]["result"] = xt::view(r_tens, 1ul, xt::all(), xt::all(), xt::all()); // peps
        jsn["algo"][prr_name]["result"] = xt::view(r_tens, 2ul, xt::all(), xt::all(), xt::all()); //rr
        jsn["meta"]["K"] = K;
        jsn["meta"]["d"] = d;
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["ninst"] = ninst;
        jsn["meta"]["seed"] = seed;
        jsn["meta"]["arms_means"] = all_arms_means;
        jsn["meta"]["min_cplx"] = min_cplx;
        jsn["meta"]["max_cplx"] = max_cplx;
        jsn["meta"]["cov"] = cov;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}

namespace xp::main::bayesian::bernoulli{
    /*
     * run experiment on the cov_boost instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform, Oracle]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness & size_of_ps of each algo]
     */
    static dt delta = 0.01f;
    static size_t ninst= 500; // number of instances to generate
    static size_t K = 8;
    static size_t d = 2;
    static dt min_cplx = 100.;
    static dt max_cplx = 500.;
    static std::string out_rep = "../out/main/bayes/";
    static std::string f_name = "mbayes_bernoulli.json";
    static size_t seed = 42ul;
    static const vv<dt,2> cov = 0.5 * xt::eye<dt>(d); // covariance is fixed and we make mu vary
    vv<size_t ,3ul> r_tens;
    vv<dt,3> all_arms_means;
    vv<dt,2> arms_means;
    std::string ppsi_name, pape_name, prr_name;
    inline int run(){
        ada_hedge adh{K};
        r_tens = xt::empty<dt>({3ul,ninst, 3ul});
        all_arms_means = xt::empty<dt>({ninst,K, d});
        size_t ctr = 0; // number of instances already executed
        dt cplx;
        size_t p;
        bool do_once = false;
        do {
            // generate parameter randomly
            // check if the complexity is inside the good range
            do {
                arms_means = xt::random::rand<dt>({K, d}, 0, 1);
                cplx = utils::psi::compute_cplxty(arms_means);
            } while(cplx>max_cplx || cplx <min_cplx);
            std::cout<<"instance found "<<cplx<<"\n";
            //std::cout<<arms_means<<"\n";
            // define bandit and run algos
            struct bernoulli bern(arms_means, cov);
            p = bern.ps.size();
            // define bandit algos
            // [APE, PEPS, Uniform]
            peps_psi ppsi(bern,adh);
            ape pape(bern);
            rr prr(bern);
            if (!do_once){
                // save the names
                pape_name = pape.name;
                ppsi_name = ppsi.name;
                prr_name = prr.name;
                do_once = true; }
            exec_algo(0ul, pape);
            exec_algo(1ul, ppsi);
            exec_algo(2ul, prr);
            xt::view(all_arms_means, ctr, xt::all(), xt::all()) = arms_means;
            ++ctr;
        } while (ctr < ninst);
        // parse and save the data
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json ape
        jsn["algo"][pape_name]["result"] = xt::view(r_tens, 0ul, xt::all(), xt::all(), xt::all());
        jsn["algo"][ppsi_name]["result"] = xt::view(r_tens, 1ul, xt::all(), xt::all(), xt::all()); // peps
        jsn["algo"][prr_name]["result"] = xt::view(r_tens, 2ul, xt::all(), xt::all(), xt::all()); //rr
        jsn["meta"]["K"] = K;
        jsn["meta"]["d"] = d;
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["ninst"] = ninst;
        jsn["meta"]["seed"] = seed;
        jsn["meta"]["arms_means"] = all_arms_means;
        jsn["meta"]["min_cplx"] = min_cplx;
        jsn["meta"]["max_cplx"] = max_cplx;
        jsn["meta"]["cov"] = cov;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}
//experience to plot K vs stopping time for the algo
#define generate_psi_instance(K, u_init, u_add) \
[&K](){                          \
vv<dt,2> xr = xt::empty<dt>({K, (u_init).size()}); \
xt::row(xr, 0ul) = u_init;                                                 \
for (size_t i=1; i<K; ++i){                     \
xt::row(xr, i) = xt::row(xr, (size_t)((int)i-1)) + (u_add);}               \
return xr; }()
namespace xp::main::num_arms_vs_tau{
    /*
     * run experiment to plot number of arms vs stopping time
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness & size_of_ps of each algo]
     */
    static dt delta = 0.01f;
    static size_t nruns = 250; // number of runs
    static std::string out_rep = "../out/main/num_arms_vs_tau/";
    static std::string f_name = "mnum_arms_vs_tau.json";
    static size_t seed = 42ul;
    static size_t d = 2;
    static vv<size_t> K_range = xt::arange(10, 20, 2);
    static vv<dt> u_init = xt::random::rand<dt>({d});
    static vv<dt> u_add = {-0.25, 0.25};
    static vv<dt,2> cov = 0.5 * xt::eye<dt>(d); // covariance is fixed and we make mu vary
    vv<size_t,4ul> r_tens;
    bool do_once{false};
    std::string ppsi_name, pape_name, prr_name;
    inline int run(){
        size_t K;
        r_tens = xt::empty<dt>({3ul, K_range.size(), nruns, 2ul});
        // generate seeds
        vv<size_t> seeds = xt::arange<size_t>(nruns);
        for(size_t i=0; i<K_range.size(); ++i){
            K=K_range(i);
            // define bandit and run algos
            auto means = generate_psi_instance(K,u_init, u_add);
            gaussian gauss(means, cov);
            //define algorithms
            ada_hedge adh{K};
            peps_psi ppsi(gauss,adh);
            ape pape(gauss);
            rr prr(gauss);
            // batch run each algo
            xt::view(r_tens, 0ul, i, xt::all(), xt::all()) = batch_ape(pape,gauss,delta,seeds);
            xt::view(r_tens, 1ul, i, xt::all(), xt::all()) = batch_peps_psi(ppsi,gauss,delta,seeds);
            xt::view(r_tens, 2ul, i, xt::all(), xt::all()) = batch_rr(prr,gauss,delta,seeds);
            if (!do_once){
                // save the names
                pape_name = pape.name;
                ppsi_name = ppsi.name;
                prr_name = prr.name;
                do_once = true; }
        }
        // parse and save the data
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json ape
        jsn["algo"][pape_name]["result"] = xt::view(r_tens, 0ul, xt::all(), xt::all(), xt::all()); //ape
        jsn["algo"][ppsi_name]["result"] = xt::view(r_tens, 1ul, xt::all(), xt::all(), xt::all()); // peps
        jsn["algo"][prr_name]["result"] = xt::view(r_tens, 2ul, xt::all(), xt::all(), xt::all()); //rr
        jsn["meta"]["K_range"] = K_range;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}
// we evaluate the impact of correlation in a particular instance
namespace xp::main::correlation{
    /*
    * run experiment to plot sample complexity vs correlation coefficient
    * @parameters [delta, Sigma]
    * @Algos [PEPS, APE]
    * @Max learner[AdaHedge]
    * @Save [SC@correctness of each algo]
    */
    static dt delta = 0.001f;
    static size_t nruns = 1000; // number of runs
    static size_t numvals = 20; // number of values of alpha
    static std::string out_rep = "../out/main/correlation/";
    static std::string f_name = "mcorrelation.json";
    static const vv<dt,2> means = CORRELATION_MEANS;
    static vv<dt> correl_coeffs = xt::concatenate(xt::xtuple(xt::linspace<dt>(-1.+1e-3, 1.-1e-3, numvals), vv<dt>{0.}));
    static vv<dt,2> cov = xt::eye<dt>(means.shape()[1]); // the off-diagonal terms will vary
    vv<size_t, 4ul> r_tens;
    bool do_once{false};
    std::string ppsi_name, pape_name;
    inline int run(){
        dt c_coeff;
        r_tens = xt::empty<dt>({2ul, numvals+1, nruns, 2ul});
        // generate seeds
        vv<size_t> seeds = xt::arange<size_t>(nruns);
        // last iteration corresponds to totally uncorrelated
        for(size_t i=0; i<numvals+1; ++i){
            c_coeff = correl_coeffs(i);
            cov(0, 1) = c_coeff;
            cov(1, 0) = c_coeff;
            std::cout<<i/(dt)(numvals+1)<<"\n";
            // define bandit and run algos
            gaussian gauss(means, cov);
            //define algorithms
            ada_hedge adh{means.shape()[0]};
            peps_psi ppsi(gauss,adh);
            ape pape(gauss);
            // batch run each algo
            xt::view(r_tens, 0ul, i, xt::all(), xt::all()) = batch_ape(pape,gauss,delta,seeds);
            xt::view(r_tens, 1ul, i, xt::all(), xt::all()) = batch_peps_psi(ppsi,gauss,delta,seeds);
            if (!do_once){
                // save the names
                pape_name = pape.name;
                ppsi_name = ppsi.name;
                do_once = true; }
        }
        // parse and save the data
        json jsn;  // output json
        std::ofstream of(out_rep + f_name); // output file
        // saving into json ape
        jsn["algo"][pape_name]["result"] = xt::view(r_tens, 0ul, xt::all(), xt::all(), xt::all()); //ape
        jsn["algo"][ppsi_name]["result"] = xt::view(r_tens, 1ul, xt::all(), xt::all(), xt::all()); // peps
        jsn["meta"]["correl_coeff"] = correl_coeffs;
        jsn["meta"]["arms_means"] = means;
        jsn["meta"]["delta"] = delta;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}
namespace xp::main::covboost::fb{
    /*
     * run experiment on a given instance to measure the timestep duration
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE]
     * @Max learner[AdaHedge]
     * @Save [time, correctness and M(for peps)]
     * RUN IN SEQUENTIAL TO AVOID BORDER EFFECTS
     */
    static dt delta = 0.1f; // confidence level
    static size_t niter = 1000ul; // number of iterations
    static size_t T = 5000ul;
    static std::string out_rep = "../out/main/cov_boost/";
    static std::string f_name = "mcovboost_fb.json";

    inline int run() {
        gaussian gauss(COV_BOOST_MEANS,COV_BOOST_COV);
        ada_hedge adh{COV_BOOST_K};
        peps_psi ppsi{gauss, adh};
        ape pape{gauss};
        // generate seeds
        vv<size_t> seeds = xt::arange<size_t>(niter);
        vv<dt,3> res_ape = xt::empty<dt>({niter, T-COV_BOOST_K, 6ul});
        vv<dt,3> res_peps = xt::empty<dt>({niter, T-COV_BOOST_K, 6ul});
        for (size_t i = 0; i < niter; ++i) {
            xt::view(res_ape, i, xt::all(), xt::all()) = fun_ape(seeds(i), gauss, T, T, delta, T);
            xt::view(res_peps, i, xt::all(), xt::all()) = fun_peps(seeds(i), gauss, adh, T, T, delta, T);
        }
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json peps @ ape
        jsn["algo"][ppsi.name]["result"] = res_peps;
        jsn["algo"]["RR"]["result"] = res_ape;
        jsn["meta"]["K"] = COV_BOOST_K;
        jsn["meta"]["delta"] = delta;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
    // save the data to json file
}

// bayesian experiment in the main
namespace xp::appx::bayesian::gaussian{
    /*
     * run experiment on the cov_boost instance
     * @parameters [delta, Sigma]
     * @Algos [PEPS, APE, Uniform, Oracle]
     * @Max learner[AdaHedge]
     * @Save [SC@correctness & size_of_ps of each algo]
     */
    static dt delta = 0.01f;
    static size_t ninst= 100; // number of instances to generate
    static size_t K = 15;
    static size_t d = 2;
    static dt min_cplx = 100.;
    static dt max_cplx = 500.;
    static std::string out_rep = "../out/appx/bayes/";
    static std::string f_name = "abayes_bern_d2_K15.json";
    static size_t seed = 42ul;
    static vv<dt,2> cov = 0.25 * xt::eye<dt>(d); // covariance is fixed and we make mu vary
    vv<size_t,3ul> r_tens;
    vv<dt,3> all_arms_means;
    vv<dt,2> arms_means;
    std::string ppsi_name, pape_name, prr_name;
    inline int run(){
        ada_hedge adh{K};
        r_tens = xt::empty<size_t>({3ul,ninst, 3ul});
        all_arms_means = xt::empty<dt>({ninst,K, d});
        size_t ctr = 0; // number of instances already executed
        dt cplx;
        size_t p;
        bool do_once = false;
        do {
            // generate parameter randomly
            // check if the complexity is inside the good range
            do {
                arms_means = xt::random::rand<dt>({K, d}, 0, 1);
                cplx = utils::psi::compute_cplxty(arms_means);
            } while(cplx>max_cplx || cplx <min_cplx);
            std::cout<< "number: " <<ctr<< " complexity: "<<cplx<<"\n";
            // define bandit and run algos
            //struct gaussian gauss(arms_means, cov);
            bernoulli bern(arms_means, cov);
            p = bern.ps.size();
            // define bandit algos
            // [APE, PEPS, Uniform]
            peps_psi ppsi(bern,adh);
            ape pape(bern);
            rr prr(bern);
            if (!do_once){
                // save the names
                pape_name = pape.name;
                ppsi_name = ppsi.name;
                prr_name = prr.name;
                do_once = true; }
                // ape
            ;
# define exec_algo(idx, algo) \
{ auto res = (algo).loop(seed, delta); \
    xt::view(r_tens, (idx), ctr, xt::all()) = vv<size_t>{res.first, res.second, p} ;}
            // peps
            auto _res_ = fun_peps(seed, bern, adh, 0ul, std::numeric_limits<size_t>::max(), delta, 25000ul);
            xt::view(r_tens, 1ul, ctr, xt::all()) = vv<size_t>{xt::view(_res_, -1, 2ul), _res_.shape()[0ul]+K, p };//{res.first, res.second, p} ;
            _res_ = fun_ape(seed, bern, 0ul, std::numeric_limits<size_t>::max(), delta, 25000ul);
            xt::view(r_tens, 0ul, ctr, xt::all()) = vv<size_t>{xt::view(_res_, -1, 2ul), _res_.shape()[0ul]+K, p };//{res.first, res.second, p} ;
            exec_algo(2ul, prr);
            xt::view(all_arms_means, ctr, xt::all(), xt::all()) = arms_means;
            ++ctr;

        } while (ctr < ninst);
        // parse and save the data
        json jsn;  // output json
        std::ofstream of(out_rep+f_name); // output file
        // saving into json ape
        jsn["algo"][pape_name]["result"] = xt::view(r_tens, 0ul, xt::all(), xt::all(), xt::all());
        jsn["algo"][ppsi_name]["result"] = xt::view(r_tens, 1ul, xt::all(), xt::all(), xt::all()); // peps
        jsn["algo"][prr_name]["result"] = xt::view(r_tens, 2ul, xt::all(), xt::all(), xt::all()); //rr
        jsn["meta"]["K"] = K;
        jsn["meta"]["d"] = d;
        jsn["meta"]["delta"] = delta;
        jsn["meta"]["ninst"] = ninst;
        jsn["meta"]["seed"] = seed;
        jsn["meta"]["arms_means"] = all_arms_means;
        jsn["meta"]["min_cplx"] = min_cplx;
        jsn["meta"]["max_cplx"] = max_cplx;
        jsn["meta"]["cov"] = cov;
        // dump into file and close
        // does not raise exception
        of <<jsn.dump()<<std::endl;
        of.close();
        return 0;
    }
}