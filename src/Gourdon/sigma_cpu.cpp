/// sigma_cpu.cpp
///
/// A cpu implementation of Gourdon's Sigma for debugging the gpu implementation
/// (c) 2016 Curtis Seizert <cseizert@gmail.com>

#include <stdint.h>
#include <CUDASieve/cudasieve.hpp>
#include <math.h>

#include "cudapix.hpp"
#include "Gourdon/gourdonvariant.hpp"

int64_t GourdonVariant64::sigma_cpu()
{
  int64_t s[7];
  int64_t s_tot = 0;
  s[0] = sigma0_cpu();
  s[1] = sigma1_cpu();
  s[2] = sigma2_cpu();
  s[3] = sigma3_cpu();
  s[4] = sigma4_cpu();
  s[5] = sigma5_cpu();
  s[6] = sigma6_cpu();

  for(uint16_t i = 0; i < 7; i++){
    std::cout << "Sigma " << i << " = " << s[i] << std::endl;
    s_tot += s[i];
  }

  return s_tot;
}

int64_t GourdonVariant64::sigma0_cpu()
{
  int64_t s0 = pi_y  - 1;
  s0 += (pi_sqrtx * (pi_sqrtx - 1)) / 2;
  s0 -= pi_y * (pi_y - 1) / 2;

  return s0;
}

int64_t GourdonVariant64::sigma1_cpu()
{
  int64_t s1 = (pi_y - pi_cbrtx) * (pi_y - pi_cbrtx - 1) / 2;

  return s1;
}

int64_t GourdonVariant64::sigma2_cpu()
{
  int64_t s2 = pi_cbrtx - pi_sqrtz;
  s2 -= pi_sqrtz * (pi_sqrtz - 3) / 2;
  s2 += pi_qrtx * (pi_qrtx - 3) / 2;
  s2 *= pi_y;

  return s2;
}

int64_t GourdonVariant64::sigma3_cpu()
{
  int64_t s3 = pi_cbrtx;
  s3 *= (pi_cbrtx - 1) * (2 * pi_cbrtx - 1) / 6;
  s3 -= pi_cbrtx;
  s3 -= pi_qrtx * (pi_qrtx - 1) * (2 * pi_qrtx - 1) / 6;
  s3 += pi_qrtx;

  return s3;
}

int64_t GourdonVariant64::sigma4_cpu()
{
  int64_t s4 = 0;

  PrimeArray pq(qrtx + 1, sqrtz);
  PrimeArray pi(sqrtz, x / (y * qrtx));

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  CudaSieve * sieve = new CudaSieve(0, pi.top);

  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    uint64_t s = sieve->countPrimesSegment((uint64_t)0, (uint64_t) x / (p * y));
    // std::cout << "S4 = " << s * pi_y << std::endl;
    s4 += s;
  }

  s4 *= pi_y;

  delete sieve;
  cudaDeviceReset();

  return s4;
}

int64_t GourdonVariant64::sigma5_cpu()
{
  int64_t s5 = 0;

  PrimeArray pq(sqrtz + 1, cbrtx);
  PrimeArray pi(cbrtx, y);

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  CudaSieve * sieve = new CudaSieve(0, pi.top);

  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    uint64_t s = sieve->countPrimesSegment((uint64_t)0, (uint64_t) x / (p * p));

    // std::cout << "S5 = " << s << std::endl;

    s5 += s;
  }

  delete sieve;
  cudaDeviceReset();

  return s5;
}

int64_t GourdonVariant64::sigma6_cpu()
{
  int64_t s6 = 0;

  PrimeArray pq(qrtx + 1, cbrtx);
  PrimeArray pi(cbrtx, pow(x, (double)3.0/8.0));

  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);

  CudaSieve * sieve = new CudaSieve(0, pi.top);

  for(uint32_t i = 0; i < pq.len; i++){
    uint64_t p = pq.h_primes[i];
    uint64_t s = sieve->countPrimesSegment((uint64_t)0, (uint64_t) std::sqrt(x/p));

    // std::cout << "S6 = " << s * s << std::endl;

    s6 += s * s;
  }

  delete sieve;
  cudaDeviceReset();

  return -s6;
}
