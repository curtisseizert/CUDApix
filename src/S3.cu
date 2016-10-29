#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <CUDASieve/cudasieve.hpp>

#include "sieve/phisieve_host.cuh"
#include "S3.cuh"
#include "sieve/lpf_mu.cuh"
#include "phi.cuh"

uint32_t smallPrimes[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};


uint64_t S3(uint64_t x, uint64_t y, uint32_t c)
{
  uint64_t top = x/y;
  int64_t * d_sums, s3 = 0, * h_sums;
  uint32_t * d_lpf;
  uint16_t threads = 512;
  uint32_t blocks = 1 + y/(2 * threads);
  int8_t * d_mu = gen_d_mu(0u, (uint32_t)y);
  size_t pi_y;
  uint64_t * h_primes = CudaSieve::getHostPrimes(0ull, y, pi_y, 0);

  cudaMalloc(&d_sums, y/2 * sizeof(int64_t));

  d_lpf = gen_d_lpf((uint32_t) 0, (uint32_t) y);

  Phisieve * phi = new Phisieve(y, top);

  c = 3;
  phi->firstSieve(c);

  S3_phi<<<blocks, threads>>>(x, (uint32_t)h_primes[c], y, d_sums, d_lpf, d_mu, phi->getCountDevice());
  cudaDeviceSynchronize();

  for(uint32_t a = c + 1; a < 20-1; a++){
    std::cout << h_primes[a] << std::endl;
    phi->markNext();
    phi->updateCount();

    S3_phi<<<blocks, threads>>>(x, (uint32_t)h_primes[a], y, d_sums, d_lpf, d_mu, phi->getCountDevice());
    cudaDeviceSynchronize();
  }

  h_sums = (int64_t *)malloc(y/2 * sizeof(int64_t));
  cudaMemcpy(h_sums, d_sums, y/2*sizeof(int64_t), cudaMemcpyDeviceToHost);

  s3 = thrust::reduce(thrust::device, d_sums, d_sums + y/2);

  cudaFree(d_sums);

  for(uint32_t j = 1; j < y/2; j++){
    // std::cout << x/(2*j+1) << "\t" << h_sums[j] << std::endl;
  }

  return s3;
}

__global__ void zero(int64_t * array, uint32_t max)
{
  uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = 0;
}

__global__ void S3_phi(uint64_t x, uint32_t p, uint64_t y, int64_t * d_sums,
                       uint32_t * d_lpf, int8_t * d_mu, uint32_t * d_phi)
{
  uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x; // m corrected for the arrays only holding odds
  uint32_t m = 2 * tidx +1;

  if(m <= y && m >= 1 + (y / p)){
    int8_t mu = d_mu[tidx];
    if(d_lpf[tidx] > p){
      uint64_t n = x / (p * m);
      int64_t phi = d_phi[(1 + n)/2];
      d_sums[tidx] -= phi * mu;
    }
  }
}
