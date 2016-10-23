/*
V = V_a + V_b
V_a = x^(1/4) < p

 */

#include <iostream>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cinttypes>

#include "P2.cuh"
#include "V.cuh"
#include "CUDASieve/cudasieve.hpp"

const uint16_t threadsPerBlock = 512;

int64_t V(uint64_t x, uint64_t y)
{
  int64_t sum = 0, sum_b = 0;
  PrimeArray pi, p, q; // since p is within q, p will just be used to hold values of pi

  pi.bottom = 0;
  pi.top = sqrt(x);
  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);

  p.bottom = sqrt(sqrt(x));
  p.top = sqrt(x/y);

  q.bottom = 0;
  q.top = y;
  // std::cout << p.bottom << " " << p.top << " " << q.top << std::endl;

  sum = V_a(x, pi, p, q);
  std::cout << "V_a = " << sum << std::endl;

  p.bottom = sqrt(x/y);
  p.top = std::cbrt(x);

  q.bottom = 0;
  q.top = y;

  // std::cout << p.bottom << " " << p.top << " " << q.top << std::endl;

  sum_b = V_b(x, pi, p, q);
  std::cout << "V_b = " << sum_b << std::endl;
  sum += sum_b;

  cudaFree(pi.d_primes);

  return sum;
}

int64_t V_a(uint64_t x, PrimeArray & pi, PrimeArray & p, PrimeArray & q) // the upper bounds of q is constant
{
  int64_t sum;

  q.d_primes = CudaSieve::getDevicePrimes(q.bottom, q.top, q.len, 0);  // holds all possible p, q and those below minimum p

  p.pi_bottom = CudaSieve::countPrimes(p.bottom);         // these numbers will define the bounds
  p.pi_top = CudaSieve::countPrimes(p.top);         // of p within the array q.d_primes

  uint64_t * d_quotients;
  cudaMalloc(&d_quotients, (q.len - p.pi_bottom)*sizeof(uint64_t));      // this array will hold x/(p*q)

  // for each p (all of these lie between q.d_primes[p.pi_bottom and q.d_primes[p.pi_top])
  for(uint64_t i = p.pi_bottom; i < p.pi_top; i++){
    uint64_t * qstart = q.d_primes + i;                             // move pointer up the array such that array[0] = p
    uint32_t blocks = 1 + (q.len - i)/threadsPerBlock;
    device::XoverPQ<<<blocks, threadsPerBlock>>>(x, qstart, d_quotients, (size_t)(q.len - i));   // calculate x/(p*q) for this p
    cudaDeviceSynchronize();

     // calculate pi(x/(p*q)) for this p and store the result back on top of x/(p*q)
    thrust::upper_bound(thrust::device, pi.d_primes, pi.d_primes+pi.len, d_quotients, d_quotients + (q.len - i), d_quotients);
    sum += thrust::reduce(thrust::device, d_quotients, d_quotients + (q.len - i)); // calculate sum(pi(x/p*q)) for this p
    sum += (q.len - i) * (2 - i);
    // std::cout << i << " " << q.len - i << " " << sum << std::endl;
    if((int64_t)q.len - i <= 0) break;
  }

  cudaFree(d_quotients);
  return sum;
}

int64_t V_b(uint64_t x, PrimeArray & pi, PrimeArray & p, PrimeArray & q) // the upper bounds of q is inversely proportional to p
{
  int64_t sum;
  uint64_t numQ = q.len - p.pi_top;

  q.d_primes = CudaSieve::getDevicePrimes(q.bottom, q.top, q.len, 0);  // holds all possible p, q and those below minimum p

  p.pi_bottom = CudaSieve::countPrimes(p.bottom);         // these numbers will define the bounds
  p.pi_top = CudaSieve::countPrimes(p.top);         // of p within the array q.d_primes

  uint64_t * d_quotients, *h_quotients;
  cudaMalloc(&d_quotients, (q.len - p.pi_bottom)*sizeof(uint64_t));      // this array will hold x/(p*q)
  cudaMallocHost(&h_quotients, (q.len - p.pi_bottom)*sizeof(uint64_t));  // this array will hold x/(p^2)
  uint32_t blocks = 1 + (q.len - p.pi_bottom)/threadsPerBlock;
  device::XoverPSquared<<<blocks, threadsPerBlock>>>(x, q.d_primes + p.pi_bottom, d_quotients, (size_t)q.len - p.pi_bottom);
  cudaMemcpy(h_quotients, d_quotients, (q.len - p.pi_bottom)*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // for each p (all of these lie between q.d_primes[p.pi_bottom and q.d_primes[p.pi_top])
  for(uint64_t i = p.pi_bottom; i < p.pi_top; i++){
    uint64_t * qstart = q.d_primes + i;                             // move pointer up the array such that array[0] = p
    blocks = 1 + (q.len - i)/threadsPerBlock;
    device::XoverPQ<<<blocks, threadsPerBlock>>>(x, qstart, d_quotients, (size_t)(q.len - i));   // calculate x/(p*q) for this p
    cudaDeviceSynchronize();

    numQ = thrust::upper_bound(thrust::device, q.d_primes, q.d_primes + q.len, h_quotients[i - p.pi_bottom]) - q.d_primes - i;
     // calculate pi(x/(p*q)) for this p and store the result back on top of x/(p*q)
    thrust::upper_bound(thrust::device, pi.d_primes, pi.d_primes+pi.len, d_quotients, d_quotients + numQ, d_quotients);
    sum += thrust::reduce(thrust::device, d_quotients, d_quotients + numQ); // calculate sum(pi(x/p*q)) for this p
    sum += (numQ - i) * (2 - i);
    // std::cout << i << " " << numQ << " " << h_quotients[i - p.pi_bottom] << std::endl;
    if((int64_t)q.len - i <= 0) break;
  }

  cudaFree(d_quotients);
  return sum;
}

__global__ void device::XoverPQ(uint64_t x, uint64_t * q, uint64_t * quot, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  if(tidx < len) quot[tidx] = x / (q[0] * q[tidx]);       // q[0] == p in this impelementation
}

__global__ void device::XoverPSquared(uint64_t x, uint64_t * p, uint64_t * quot, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;
  if(tidx < len) quot[tidx] = x / (p[tidx] * p[tidx]);
}
