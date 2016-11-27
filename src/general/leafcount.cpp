#include <iostream>
#include <math.h>
#include <CUDASieve/cudasieve.hpp>
#include <cuda_uint128.h>
#include <omp.h>

#include "phi.cuh"
#include "cudapix.hpp"
#include "general/tools.hpp"
#include "general/leafcount.hpp"

uint64_t leafcount::gourdon_A(uint64_t x)
{
  uint64_t sqrtx = sqrt(x);
  uint64_t cbrtx = cbrt(x);
  uint64_t qrtx = sqrt(sqrtx);
  uint64_t x38 = qrtx * sqrt(qrtx);
  uint64_t numSparse = 0, numClustered = 0;
  uint64_t maxDeltaQ = 0, pMaxDeltaQ = 0;

  double ratio;

  PrimeArray pq(qrtx, sqrt(x / qrtx));

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  for(uint32_t i = 0; i < pq.h_primes[i] < cbrtx; i++){
    uint64_t p = pq.h_primes[i];
    if(p == 0) break;
    uint64_t max =      upperBound(pq.h_primes, 0, pq.len, sqrt(x/p));
    uint64_t boundary = upperBound(pq.h_primes, 0, pq.len, x/(p * x38));
    uint64_t min =      i + 1;

    boundary = std::max(boundary, min);
    max = std::max(max, boundary);

    if(max - boundary > maxDeltaQ){
      maxDeltaQ = max - boundary;
      pMaxDeltaQ = i;
    }

    numSparse += (boundary - min);
    numClustered += (max - boundary);
  }

  ratio = (double)numClustered/(double)numSparse;

  std::cout << "Number of clustered leaves\t:\t" << numClustered << std::endl;
  std::cout << "Number of sparse leaves   \t:\t" << numSparse << std::endl;
  std::cout << "Ratio of clustered to sparse\t:\t" << ratio << std::endl;
  std::cout << "Total easy leaves         \t:\t" << numClustered + numSparse << std::endl;
  std::cout << "maxDeltaQ = " << maxDeltaQ << "\t pMaxDeltaQ = " << pMaxDeltaQ << std::endl;

  cudaFreeHost(pq.h_primes);

  return numSparse + numClustered;
}

uint128_t leafcount::gourdon_A(uint128_t x)
{
  uint64_t sqrtx = _isqrt(x);
  uint64_t cbrtx = pow(sqrtx, (double)(2.0/3.0));
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
    uint64_t min =      i + 1;

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


uint64_t leafcount::gourdon_C_simple(uint64_t x, uint64_t y)
{
  uint64_t cbrtz = cbrt(x/y);
  uint64_t qrtx = sqrt(sqrt(x));
  uint64_t sum = 0;

  // if(y < pow(x, 0.4)) return 0;

  PrimeArray pq(cbrtz, qrtx);
  PrimeArray pi(0, y);

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);

  // #pragma omp parallel for reduction(+: sum)
  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    int64_t sum1;
    sum1 = pi.len;
    sum1 -= upperBound(pi.h_primes, 0, pi.len, 1 + x / (p*p*p));
    // std::cout << i << "\t\t" << sum1 << std::endl;
    sum += sum1;
  }

  cudaFreeHost(pq.h_primes);
  cudaFreeHost(pi.h_primes);

  return sum;
}
/*
uint64_t leafcount::omega1(uint64_t x, uint64_t y, uint16_t c)
{
  uint64_t sum = 0;
  Phi * phi_ = new Phi(y, std::sqrt(y) + 1);

  PrimeArray pq(0, std::sqrt(y));

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  for(uint32_t i = c; i < pq.len - c; i++){
    sum += phi_->phi(y, i) - phi_->phi(y/pq.h_primes[i], i);
  }

  cudaDeviceReset();
  return sum;
}
*/
uint64_t leafcount::omega2(uint64_t x, uint64_t y)
{
  uint64_t sum = 0;
  PrimeArray pq(std::sqrt(y) + 1, std::sqrt(std::sqrt(x)));
  PrimeArray pi(0, y);

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);

  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    sum += upperBound(pi.h_primes, 0, pi.len, std::min(y, x/(p*p*p))) - upperBound(pi.h_primes, 0, pi.len, y/p);
  }

  cudaDeviceReset();
  return sum;
}

uint64_t leafcount::omega3(uint64_t x, uint64_t y)
{
  uint64_t sum = 0;
  PrimeArray pq(std::sqrt(y) + 1, std::sqrt(std::sqrt(x)));
  PrimeArray pi(0, y);

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);

  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    sum += pq.len - upperBound(pi.h_primes, 0, pi.len, x/(p*p*p));
  }

  cudaDeviceReset();
  return sum;
}

uint64_t leafcount::omega3(uint128_t x, uint64_t y)
{
  uint64_t sum = 0;
  PrimeArray pq(_isqrt(y) + 1, std::sqrt(_isqrt(x)));
  PrimeArray pi(0, y);

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);

  #pragma omp parallel for
  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    sum += pi.len - upperBound(pi.h_primes, 0, pi.len, x/(p*p*p));
  }

  cudaDeviceReset();
  return sum;
}
