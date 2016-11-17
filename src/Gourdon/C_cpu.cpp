// C_cpu.cpp
//
// A simple cpu implemenation of the sum "C" in Xavier Gourdon's variant of the
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

uint64_t GourdonVariant64::C_cpu()
{
  // sum(x^1/4 < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y
  uint64_t sum = 0;

  uint64_t pi_sqrty = CudaSieve::countPrimes((uint64_t)0, (uint64_t) std::sqrt(y));
  uint32_t num_p = pi_qrtx - pi_sqrty;

  uint64_t upper_bound = y;
  uint64_t lower_bound = std::sqrt(y);
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

  // we also have to find which q to start at: q = x / p^3.  We will do this by
  // creating an array and storing start positions in it (this will simplify
  // transition to a segmented version)
  uint32_t * firstQ = (uint32_t *)malloc((pi_qrtx - pi_sqrty) * sizeof(uint32_t));
  #pragma omp parallel for
  for(uint32_t i = 0; i < num_p; i++)
    firstQ[i] = upperBound(pq, 0, len, x / (pq[i] * pq[i] * pq[i])) + 1;

  // We now iterate through our list of p's and q's with p's in the outer for
  // loop and q's in the inner one.  Also using openMP because I paid for those
  // cores...
  #pragma omp parallel for schedule(dynamic) reduction(+:sum)
  for(uint32_t i = 0; i < num_p; i++){ // p values
    uint64_t p = pq[i];

    for(uint32_t j = firstQ[i]; j < len; j++){ // q values
      uint64_t q = pq[j];

      // now find pi(x /( p * q) for all q values in range
      uint64_t quot = x / (p * q);
      uint64_t pi_quot = h_pitable[(quot + 1)/2];

      // now add (2 - pi(p))
      sum += (2 - (i + pi_sqrty));

      // and pi(x/pq)
      sum += pi_quot;
    }
  }

  return sum;
}
