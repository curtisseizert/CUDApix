// A2_cpu.cpp
//
// A simple cpu implemenation of the sum "A" in Xavier Gourdon's variant of the
// Deleglise-Rivat prime counting algorithm.
//
// Copywrite (c) 2016 Curtis Seizert <cseizert@gmail.com>

#include <iostream>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include <CUDASieve/cudasieve.hpp>

#include "general/tools.hpp"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"

uint64_t GourdonVariant64::A2_cpu()
{
  // sum(x^1/4 < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y
  uint64_t sum = 0;

  uint32_t * d_pitable, * h_pitable;

  // this is the number of p values we will iterate over
  uint64_t num_p = pi_cbrtx - pi_qrtx;

  uint64_t upper_bound = pow(x, (double)0.375); // x^(3/8)
  uint64_t lower_bound = qrtx;  // x^(1/4)
  size_t len; // this will hold the number of q values

  // get a list of primes for the p and q
  uint64_t * pq = CudaSieve::getHostPrimes(lower_bound, upper_bound, len, 0);

  // make an array to hold the last q index for each p in a given segment
  uint32_t * lastQ = (uint32_t *)malloc(num_p * sizeof(uint32_t));

  // fill the last q array with the first q it will hold (e.g. pi(p) + 1)
  #pragma omp parallel for
  for(uint32_t i = 0; i < num_p; i++)
    lastQ[i] = i + 1;

  // set the bounds of the pi table (start, bottom, segment size)
  PiTable pi_table(sqrtx, upper_bound, 1u << 30);
  cudaMallocHost(&h_pitable, pi_table.get_range()/2 * sizeof(uint32_t));

  // set the value of pi(x) for the start of the pi table to save us from calculating
  // that again (it is calculated automatically with the GourdonVariant64
  // constructor)
  pi_table.set_pi_base(pi_sqrtx);

  while(pi_table.get_base() > upper_bound){

    // Get pi table for this iteration (moving downward from x^1/2 to x^1/3)
    d_pitable = pi_table.getNextDown();
    cudaMemcpy(h_pitable, d_pitable, pi_table.get_range()/2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Do the same thing as in A_cpu except set the upper bound so that we don't
    // go outside the range of the pi table
    // upper bound if defined by intersection x / p^2 = [base of pi table]
    uint64_t pMax =  std::sqrt(x / pi_table.get_base());
    uint32_t pMaxIdx = upperBound(pq, 0, num_p, pMax);

    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    for(uint32_t i = 0; i < num_p; i++){
      uint64_t p = pq[i];
      uint64_t maxQ = std::sqrt(x/p);

      for(; lastQ[i] < len; lastQ[i]++){
        uint64_t q = pq[lastQ[i]];
        if(q > maxQ) break;

        // now find pi(x /( p * q) for all q values in range
        uint64_t quot = x / (p * q);
        if(quot < pi_table.get_base()) break;
        uint64_t pi_quot = h_pitable[(quot + 1 - (pi_table.get_base() & ~1ull))/2] + pi_table.get_pi_base();

        // we now double pi_quot if quot is < y (to account for the chi term)
        pi_quot += (quot < y) ? pi_quot : 0;
        // std::cout << pi_quot << std::endl;

        sum += pi_quot;
      }
    }
  }
  std::cout << "\nSum of low PQ = " << sum << std::endl;
  uint64_t sum1 = sum;
  sum = 0;

  pi_table.set_bottom(cbrtx);

  while(pi_table.get_base() > cbrtx){

    // Get pi table for this iteration (moving downward from x^1/2 to x^1/3)
    d_pitable = pi_table.getNextDown();
    cudaMemcpy(h_pitable, d_pitable, pi_table.get_range()/2 * sizeof(uint32_t) + 16, cudaMemcpyDeviceToHost);

    // Do the same thing as in A_cpu except set the upper bound so that we don't
    // go outside the range of the pi table
    // upper bound if defined by intersection x / p^2 = [base of pi table]
    uint64_t pMax =  std::sqrt(x / pi_table.get_base());
    uint32_t pMaxIdx = upperBound(pq, 0, num_p, pMax);

    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    for(uint32_t i = 0; i < num_p; i++){
      uint64_t p = pq[i];
      uint64_t maxQ = std::sqrt(x/p);

      for(; lastQ[i] < len; lastQ[i]++){
        uint64_t q = pq[lastQ[i]];
        if(q > maxQ) break;

        // now find pi(x /( p * q) for all q values in range
        uint64_t quot = x / (p * q);
        if(quot <= pi_table.get_base()) break;
        uint64_t pi_quot = h_pitable[(quot + 1 - (pi_table.get_base() & ~1ull))/2] + pi_table.get_pi_base();

        // we now double pi_quot if quot is < y (to account for the chi term)
        pi_quot += (quot < y) ? pi_quot : 0;
        // std::cout << pi_quot << std::endl;

        sum += pi_quot;
      }
    }
  }

  std::cout << "\nSum of hi PQ = " << sum << std::endl;


  return sum + sum1;
}
