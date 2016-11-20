// A_cpu.cpp
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

uint64_t GourdonVariant64::A_cpu()
{
  // sum(x^1/4 < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y
  uint64_t sum = 0;

  // this is the number of p values we will iterate over
  uint64_t num_p = pi_cbrtx - pi_qrtx;

  uint64_t upper_bound = pow(x, (double)0.375); // x^(3/8)
  uint64_t lower_bound = qrtx;  // x^(1/4)
  size_t len; // this will hold the number of q values

  // get a list of primes for the p and q
  uint64_t * pq = CudaSieve::getHostPrimes(lower_bound, upper_bound, len, 0);

  // generate a pi table up to sqrtx on the device (this can be changed so it's
  // all done on the cpu, but this is easier for me since I have all the code in
  // place for pi table)
  PiTable * pi_table = new PiTable(sqrtx);
  uint32_t * d_pitable = pi_table->getCurrent();


  // bring the pi table back to the host
  uint32_t * h_pitable = (uint32_t *)malloc(sqrtx / 2 * sizeof(uint32_t));
  cudaMemcpy(h_pitable, d_pitable, sqrtx / 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // We now iterate through our list of p's and q's with p's in the outer for
  // loop and q's in the inner one.  Also using openMP because I paid for those
  // cores...
  // #pragma omp parallel for schedule(dynamic) reduction(+:sum)
  for(uint32_t i =0; i < num_p; i++){ // p values
    uint64_t p = pq[i];
    uint64_t maxQ = /*pow(x, (double) 5.0/8.0)/p - 1;*/std::sqrt(x/p); // upper bound for q

    for(uint32_t j = i + 1; pq[j] <= maxQ; j++){ // q values
      uint64_t q = pq[j];
      if(q == 0) break; // just in case it goes past maxQ

      // now find pi(x /( p * q) for all q values in range
      uint64_t quot = x / (p * q);
      uint64_t pi_quot = h_pitable[(quot + 1)/2];

      // we now double pi_quot if quot is < y (to account for the chi term)
      pi_quot += (quot < y) ? pi_quot : 0;
      // std::cout << pi_quot << std::endl;

      sum += pi_quot;
    }
  }

  return sum;
}
