#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "S3.cuh"
#include "sieve/lpf_mu.cuh"
#include "phi.cuh"

uint32_t smallPrimes[13] = {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

uint64_t S3(uint64_t x, uint64_t y, uint32_t c)
{
  uint64_t top = x/y;
  int64_t * d_sums, s3 = 0;
  uint32_t maxPrime = (uint32_t) sqrt(sqrt(x));
  uint32_t * h_primeList, * d_lpf;
  uint16_t threads = 512;
  uint32_t blocks = 1 + y/(2 * threads);
  int8_t * d_mu = gen_d_mu(0u, (uint32_t)y);

  Phi phi((uint32_t)top , maxPrime);

  h_primeList = phi.get_h_primeList();
  d_lpf = gen_d_lpf(0u, (uint32_t)y);

  cudaMalloc(&d_sums, y * sizeof(int64_t)/2);
  cudaMemset(d_sums, y * sizeof(int64_t)/2, 0);

  zero<<<blocks, threads>>>(d_sums, y/2);
  cudaDeviceSynchronize();

  for(uint32_t i = c; i < phi.get_primeListLength(); i++){
    uint32_t * d_phi = phi.generateRange(top, i-1);
    uint32_t p;

    if(i > 12) p = h_primeList[i - 13];
    else p = smallPrimes[i];

    S3_phi<<<blocks, threads>>>(x, p, y, d_sums, d_lpf, d_mu, d_phi);
    cudaDeviceSynchronize();
    cudaFree(d_phi);
  }

  s3 = thrust::reduce(thrust::device, d_sums, d_sums + y/2);

  return s3;
}

__global__ void zero(int64_t * array, uint32_t max)
{
  uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = 0;
}


__global__ void S3_phi(uint64_t x, uint32_t p, uint64_t y, int64_t * d_sums, uint32_t * d_lpf, int8_t * d_mu, uint32_t * d_phi)
{
  uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x; // m corrected for the arrays only holding odds
  uint32_t m = 2 * tidx + 1;

  if(m <= y && m >= y/p){
    int8_t mu = d_mu[tidx+ 1];
    if(d_lpf[tidx+1] > p && mu != 0){
      uint32_t n = x / (p * m);
      int64_t phi = 1 + d_phi[n/2];
      d_sums[tidx] -= mu * phi;
    }
  }
}
