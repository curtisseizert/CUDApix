#include <cuda.h>
#include <math_functions.h>
#include <stdint.h>
#include <stdio.h>

#include "sieve/phisieve_constants.cu"
#include "sieve/phisieve_device.cuh"

__device__
void phidev::sieveCountInit(uint32_t * d_sieve, uint32_t * d_count,
                            uint64_t bstart, uint16_t c)
{
  uint64_t j = bstart/64 + threadIdx.x + blockIdx.x*blockDim.x;
  uint32_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  uint32_t s = 0;
  s = d_bitmask[0][j % d_smallPrimes[0]];
  for(uint16_t a = 1; a < c - 1; a++)
    s |= d_bitmask[a][j % d_smallPrimes[a]];
  d_sieve[tidx] = s;

  #pragma unroll 32
  for(uint16_t i = 0; i < 32; i++)
    d_count[32 * tidx + i] = 1u & ~(s >> i);
}

__device__
void phidev::markSmallPrimes(uint32_t * d_sieve, uint64_t bstart, uint16_t c)
{
  uint64_t j = bstart/64 + threadIdx.x + blockIdx.x*blockDim.x;
  uint32_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  d_sieve[tidx] |= d_bitmask[c - 1][j % d_smallPrimes[c - 1]];
}

__device__
void phidev::markMedPrimes(uint32_t * d_sieve, uint32_t p, uint64_t bstart,
                           uint32_t sieveBits)
{
  bstart += 2 * (threadIdx.x + blockIdx.x * blockDim.x) * sieveBits;
  uint32_t off = p - bstart % p;
  if(off%2==0) off += p;
  off = off >> 1; // convert offset to align with half d_sieve
  for(; off < sieveBits; off += p) atomicOr(&d_sieve[(bstart >> 1) + (off >> 5)], (1u << (off & 31)));
}

__global__
void phiglobal::sieveCountInit(uint32_t * d_sieve, uint32_t * d_count,
                               uint64_t bstart, uint16_t c)

{
  phidev::sieveCountInit(d_sieve, d_count, bstart, c);
  __syncthreads();
}

__global__
void phiglobal::markSmallPrimes(uint32_t * d_sieve, uint64_t bstart, uint16_t c)
{
  phidev::markSmallPrimes(d_sieve, bstart, c);
  __syncthreads();
}

__global__
void phiglobal::markMedPrimes(uint32_t * d_sieve, uint32_t p, uint64_t bstart,
                              uint32_t sieveBits)
{
  phidev::markMedPrimes(d_sieve, p, bstart, sieveBits);
  __syncthreads();
}

__global__
void phiglobal::updateCount(uint32_t * d_sieve, uint32_t * d_count)
{
  uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t s = d_sieve[tidx];

  #pragma unroll 32
  for(uint16_t i = 0; i < 32; i++)
    d_count[32 * tidx + i] = 1u & ~(s >> i);
}
