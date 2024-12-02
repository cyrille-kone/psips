#pragma once
#include "utils.hpp"
#include "bandits.hpp"
#include "learners.hpp"

/*
 * functional implementation of peps for psi
 */
vv<size_t,2ul> fun_peps(size_t seed,
                  bandit& b_ref,   // bandit instance
                  learner& l_ref, // max learner
                  size_t t_min=0ul, // forced  minimum number of iterations
                  size_t t_max=std::numeric_limits<size_t>::max(),// maximum number of iterations
                  dt delta=0.1f,    // confidence level
                  size_t m_alloc=0ul, // minimum pre-allocation for the vector of returns
                  dt alpha =1.f // value of $\alpha$ for forced exploration
);
/*
 * functional implementation of ape for psi
 */
vv<size_t,2ul> fun_ape(size_t seed,
                  bandit& b_ref,   // bandit instance
                  size_t t_min=0ul, // forced  minimum number of iterations
                  size_t t_max=std::numeric_limits<size_t>::max(),// maximum number of iterations
                  dt delta=0.1f,    // confidence level
                  size_t m_alloc=0ul // minimum pre-allocation for the vector of returns
);