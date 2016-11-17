#include <stdint.h>
#include <CUDASieve/cudasieve.hpp>
#include <uint128_t.cuh>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <math.h>
#include <math_functions.h>

#include "cudapix.hpp"
#include "general/device_functions.cuh"
#include "Gourdon/gourdonvariant.hpp"

const uint16_t threadsPerBlock = 256;

int64_t GourdonVariant64::sigma()
{
  int64_t s[7];
  int64_t s_tot = 0;
  s[0] = sigma0();
  s[1] = sigma1();
  s[2] = sigma2();
  s[3] = sigma3();
  s[4] = sigma4();
  s[5] = sigma5();
  s[6] = sigma6();

  for(uint16_t i = 0; i < 7; i++){
    std::cout << "Sigma " << i << " = " << s[i] << std::endl;
    s_tot += s[i];
  }

  return s_tot;
}

int64_t GourdonVariant64::sigma0()
{
  int64_t s0 = pi_y  - 1;
  s0 += (pi_sqrtx * (pi_sqrtx - 1)) / 2;
  s0 -= pi_y * (pi_y - 1) / 2;

  return s0;
}

int64_t GourdonVariant64::sigma1()
{
  int64_t s1 = (pi_y - pi_cbrtx) * (pi_y - pi_cbrtx - 1) / 2;

  return s1;
}

int64_t GourdonVariant64::sigma2()
{
  int64_t s2 = pi_cbrtx - pi_sqrtz;
  s2 -= pi_sqrtz * (pi_sqrtz - 3) / 2;
  s2 += pi_qrtx * (pi_qrtx - 3) / 2;
  s2 *= pi_y;

  return s2;
}

int64_t GourdonVariant64::sigma3()
{
  int64_t s3 = pi_cbrtx;
  s3 *= (pi_cbrtx - 1) * (2 * pi_cbrtx - 1) / 6;
  s3 -= pi_cbrtx;
  s3 -= pi_qrtx * (pi_qrtx - 1) * (2 * pi_qrtx - 1) / 6;
  s3 += pi_qrtx;

  return s3;
}

int64_t GourdonVariant64::sigma4()
{
  int64_t s4 = 0;
  PrimeArray p(qrtx + 1, sqrtz);
  PrimeArray pi(sqrtz, x / (y * qrtx));

  p.d_primes = CudaSieve::getDevicePrimes(p.bottom, p.top, p.len, 0);
  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);

  xOverPY(p.d_primes, x, y, p.len);
  cudaDeviceSynchronize();

  thrust::upper_bound(thrust::device, pi.d_primes, pi.d_primes + pi.len, p.d_primes, p.d_primes + p.len, p.d_primes);

  s4 = thrust::reduce(thrust::device, p.d_primes, p.d_primes + p.len);

  s4 += pi_sqrtz * p.len;

  s4 *= pi_y;

  return s4;
}

int64_t GourdonVariant64::sigma5()
{
  int64_t s5 = 0;
  PrimeArray p(sqrtz + 1, cbrtx);
  PrimeArray pi(cbrtx, y);

  p.d_primes = CudaSieve::getDevicePrimes(p.bottom, p.top, p.len, 0);
  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);

  xOverPSquared(p.d_primes, x, p.len);
  cudaDeviceSynchronize();

  thrust::upper_bound(thrust::device, pi.d_primes, pi.d_primes + pi.len, p.d_primes, p.d_primes + p.len, p.d_primes);

  s5 = thrust::reduce(thrust::device, p.d_primes, p.d_primes + p.len);
  s5 += p.len * pi_cbrtx;

  return s5;
}

int64_t GourdonVariant64::sigma6()
{
  int64_t s6 = 0;
  PrimeArray p(qrtx + 1, cbrtx);
  PrimeArray pi(cbrtx, pow(x, (double) 3.0/8.0));

  p.d_primes = CudaSieve::getDevicePrimes(p.bottom, p.top, p.len, 0);
  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);

  sqrtxOverSqrtp(p.d_primes, sqrtx, p.len);
  cudaDeviceSynchronize();

  thrust::upper_bound(thrust::device, pi.d_primes, pi.d_primes + pi.len, p.d_primes, p.d_primes + p.len, p.d_primes);

  addToArray(p.d_primes, p.len, pi_cbrtx);

  squareEach(p.d_primes, p.len);
  cudaDeviceSynchronize();

  s6 = thrust::reduce(thrust::device, p.d_primes, p.d_primes + p.len);

  return -s6;
}


///
/// The __global__ functions used below are defined in general/device_functions.cu
///


///  For sigma4:
///  array[i] = x / (array[i] * y);
///  which represents the equation n = x (p_i * y)
///
inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len)
{
  global::xOverPY<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, y, len);
}

inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len)
{
  global::xOverPY<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, y, len);
}


///  For sigma5:
///  array[i] = x / (array[i] * array[i]);
///  which represents the equation n = x / (p_i * p_i)
///
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len)
{
  global::xOverPSquared<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, len);
}

inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len)
{
  global::xOverPSquared<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, len);
}


///  For sigma6:
///  array[i] = sqrt(x) / sqrt(p);
///  which represents the equation n = x^(1/2) / p_i^(1/2)
///  note that sqrt(x) is precomputed, so don't call this function x.
///

inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len)
{
  global::sqrtxOverSqrtp<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, sqrtx, len);
}


///  For sigma6:
///  array[i] = array[i]^2;
///  which represents the expression pi(n)^2
///

inline void addToArray(uint64_t * pi, size_t len, uint64_t k)
{
  global::addToArray<<<len/threadsPerBlock + 1, threadsPerBlock>>>(pi, len, k);
}

inline void squareEach(uint64_t * pi, size_t len)
{
  global::squareEach<<<len/threadsPerBlock + 1, threadsPerBlock>>>(pi, len);
}
