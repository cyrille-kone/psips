#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmasked_view.hpp>
#include "src/bandits.hpp"
#include "src/policies.hpp"
#include "src/xp.hpp"
#include "src/learners.hpp"
dt delta = 0.01;
int main(){
    vv<dt,2> means = COV_BOOST_MEANS;
    vv<dt,2> cov = COV_BOOST_COV;
    // define bandit instance
    gaussian gauss (means, cov);
    // define learner
    ada_hedge hedge(means.shape()[0]);
    // define algorithms
    peps_psi ppsi(gauss, hedge);
    ape pape(gauss);
    rr prr(gauss);
    oracle porcl(gauss, COV_BOOST_W_STAR);

    // define seeds
    auto rpt = 10ul; // number of exp
    vv <size_t> seeds = xt::arange<size_t>(rpt);
    auto batch_peps_res = batch_peps_psi(ppsi, gauss, delta, seeds);
    auto batch_rr_res = batch_rr(prr, gauss, delta, seeds);
    auto batch_ape_res = batch_ape(pape, gauss, delta, seeds);
    auto batch_orcl_res = batch_oracle(porcl, gauss, delta, seeds);

    std::cout<<"Oracle: "<<xt::mean(batch_orcl_res, {0})<<"\n";
    std::cout<< "APE: "<<xt::mean(batch_ape_res, {0})<<"\n";
    std::cout<<"PEPS: "<< xt::mean(batch_peps_res, {0})<<"\n";
    std::cout<<"RR: "<< xt::mean(batch_rr_res, {0})<<"\n";


    return 0;
}
/*
 To reproduce an experiment
 the line should be uncommented and pasted
 in the body of the main function
 Refer to the header xp.hpp for the parameters of the experiments
xp::main::bernoulli::run();
xp::main::chrono::covboost::run();
xp::main::bernoulli::run();
xp::main::cov_boost::run();
xp::main::num_arms_vs_tau::run();
xp::main::correlation::run();
xp::main::bayesian::bernoulli::run();
xp::main::chrono::empM::run();
xp::main::noc::run();
xp::main::covboost::fb::run();
xp::appx::bayesian::gaussian::run();
*/
