#include <iostream>
#include <math.h>
#include <CUDASieve/cudasieve.hpp>

#include "cudapix.hpp"
#include "general/tools.hpp"
#include "general/leafcount.hpp"

uint64_t countEasyGourdon(uint64_t x)
{
  uint64_t sqrtx = sqrt(x);
  uint64_t cbrtx = cbrt(x);
  uint64_t qrtx = sqrt(sqrtx);
  uint64_t x38 = qrtx * sqrt(qrtx);
  uint64_t numSparse = 0, numClustered = 0;

  double ratio;

  PrimeArray pq(qrtx, sqrt(x / qrtx));

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  for(uint32_t i = 0; i < pq.h_primes[i] < cbrtx; i++){
    uint64_t p = pq.h_primes[i];
    if(p == 0) break;
    uint64_t max =      upperBound(pq.h_primes, 0, pq.len, sqrt(x/p));
    uint64_t boundary = upperBound(pq.h_primes, 0, pq.len, x/(p * x38));
    uint64_t min =      i;

    boundary = std::max(boundary, min);
    max = std::max(max, boundary);

    numSparse += (boundary - min);
    numClustered += (max - boundary);
  }

  ratio = (double)numClustered/(double)numSparse;

  std::cout << "Number of clustered leaves\t:\t" << numClustered << std::endl;
  std::cout << "Number of sparse leaves   \t:\t" << numSparse << std::endl;
  std::cout << "Ratio of clustered to sparse\t:\t" << ratio << std::endl;
  std::cout << "Total easy leaves         \t:\t" << numClustered + numSparse << std::endl;

  cudaFreeHost(pq.h_primes);

  return numSparse + numClustered;
}
