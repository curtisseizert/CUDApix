#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <vector>

#include <CUDASieve/cudasieve.hpp>
#include <uint128_t.cuh>

#include "sieve/phisieve_host.cuh"
#include "phi.cuh"
#include "ordinary.cuh"
#include "sieve/lpf_mu.cuh"

__constant__ uint16_t d_small[8] = {2, 3, 5, 7, 11, 13, 17, 19};
__constant__ uint32_t d_wheel[8] = {2, 6, 30, 210, 2310, 30030, 510510, 9699690};
__constant__ uint32_t d_totient[8] = {1, 2, 8, 48, 480, 5760, 92160, 1658880};

uint64_t Ordinary(uint64_t x, uint64_t y, uint16_t c)
{
  c--;
  uint16_t threads = 256;
  uint64_t sum = 0;
  int64_t * d_quot = NULL;
  int8_t * d_mu;
  uint32_t arraySize = 1+ y/2;
  uint32_t * d_phi, * d_lpf;

  d_mu = gen_d_mu(0u, (uint32_t)y);
  d_lpf = gen_d_lpf(0u, (uint32_t)y);

  Phisieve * phi = new Phisieve(1000);

  phi->firstSieve(c + 1);

  d_phi = phi->getCountDevice();

  cudaMalloc(&d_quot, arraySize * sizeof(int64_t));
  cudaMemset(d_quot, arraySize * sizeof(int64_t), 0);

  ordinaryKernel<<<1 + arraySize/threads, threads>>>(d_mu, d_quot, d_lpf, d_phi, x, y, c);

  cudaDeviceSynchronize();
  sum += thrust::reduce(thrust::device, d_quot, d_quot + (y / 2));

  cudaFree(d_mu);
  cudaFree(d_quot);
  cudaFree(d_lpf);

  delete phi;

  return sum;
}

uint128_t Ordinary(uint128_t x, uint64_t y, uint16_t c)
{
  c--;
  uint16_t threads = 256;
  uint128_t sum = 0;
  int64_t * d_quot = NULL;
  int8_t * d_mu;
  uint32_t arraySize = 1+ y/2;
  uint32_t * d_phi, * d_lpf;

  d_mu = gen_d_mu(0u, (uint32_t)y);
  d_lpf = gen_d_lpf(0u, (uint32_t)y);

  Phisieve * phi = new Phisieve(1000);

  phi->firstSieve(c + 1);

  d_phi = phi->getCountDevice();

  cudaMalloc(&d_quot, arraySize * sizeof(int64_t));
  cudaMemset(d_quot, arraySize * sizeof(int64_t), 0);

  ordinaryKernel<<<1 + arraySize/threads, threads>>>(d_mu, d_quot, d_lpf, d_phi, x, y, c);

  cudaDeviceSynchronize();
  sum += thrust::reduce(thrust::device, d_quot, d_quot + (y / 2));

  cudaFree(d_mu);
  cudaFree(d_quot);
  cudaFree(d_lpf);

  delete phi;

  return sum;
}

__global__ void ordinaryKernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint64_t x, uint64_t y, uint16_t c)
{
  uint64_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t n = 2 * tidx + 1;
  if(n <= y){
    d_quot[tidx] = 0;
    if(d_lpf[tidx] > d_small[c]){
      uint64_t m = (x / n);
      uint64_t phi = m / d_wheel[c];
      phi *= d_totient[c];
      phi += d_phi[((1+m%d_wheel[c]))/2];
      d_quot[tidx] = d_mu[tidx] * phi;
    }
  }
}

// 128 bit -- in progress
__global__ void ordinaryKernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint128_t x, uint64_t y, uint16_t c)
{
  uint64_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t n = 2 * tidx + 1;
  if(n <= y){
    d_quot[tidx] = 0;
    if(d_lpf[tidx] > d_small[c]){
      uint64_t m = (x / n);
      uint64_t phi = m / d_wheel[c];
      phi *= d_totient[c];
      phi += d_phi[((1+m%d_wheel[c]))/2];
      d_quot[tidx] = d_mu[tidx] * phi;
    }
  }
}
