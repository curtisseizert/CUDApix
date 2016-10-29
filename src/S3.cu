#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "S3.cuh"
#include "sieve/lpf_mu.cuh"
#include "phi.cuh"

uint32_t smallPrimes[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};


uint64_t S3(uint64_t x, uint64_t y, uint32_t c)
{
  uint64_t top = x/y;
  int64_t * d_sums, s3 = 0;
  uint32_t maxPrime = (uint32_t) y;
  uint32_t * h_primeList, * d_lpf;
  uint16_t threads = 512;
  uint32_t blocks = 1 + y/(2 * threads);
  int8_t * d_mu = gen_d_mu(0u, (uint32_t)y);
  uint32_t num = 0;

  Phi phi((uint32_t)top , maxPrime);

  h_primeList = phi.get_h_primeList();
  d_lpf = gen_d_lpf(0u, (uint32_t)y);
  uint32_t * h_lpf = (uint32_t *)malloc(y/2 * sizeof(uint32_t));
  int8_t * h_mu = (int8_t *)malloc(y/2);

  cudaMemcpy(h_lpf, d_lpf, y/2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mu, d_mu, y/2, cudaMemcpyDeviceToHost);

  cudaMalloc(&d_sums, y * sizeof(int64_t)/2);

  zero<<<blocks, threads>>>(d_sums, y/2);
  cudaDeviceSynchronize();

  // for(uint32_t i = c + 1; i < 100 + 2; i++){//phi.get_primeListLength() + 11; i++){
  //   num++;
  //   uint32_t * d_phi = phi.generateRange(top, i-2);
  //   uint32_t p;
  //
  //   if(i >= 13) p = h_primeList[i - 13];
  //   else p = smallPrimes[i-1];
  //
  //   S3_phi<<<blocks, threads>>>(x, p, y, d_sums, d_lpf, d_mu, d_phi);
  //   cudaDeviceSynchronize();
  //   cudaFree(d_phi);
  // }

  int64_t h_sum = 0, * h_sums = (int64_t *)malloc(y/2 * sizeof(int64_t));

  for(uint32_t j = 0; j < y/2; j++) h_sums[j] = 0;

  for(uint32_t i  = c + 1; i < 100 + 2; i++){
    uint32_t p;
    if(i >= 13) p = h_primeList[i - 13];
    else p = smallPrimes[i - 1];
    std::cout << p << std::endl;
    for(uint32_t m = (y / p) + 1 - (y / p) % 2; m <= y; m += 2){
      if(h_lpf[m/2] > p){
         h_sums[m/2] -= h_mu[m/2] * phi.phi(x / (m * p), i - 2);
      }
    }
  }

  // cudaMemcpy(h_sums, d_sums, y/2 * sizeof(int64_t), cudaMemcpyDeviceToHost);

  for(uint32_t j = 1; j < y/2; j++){
    std::cout << (x / (2 * j + 1)) << "\t" << h_sums[j] << std::endl;
    h_sum += h_sums[j];
  }

  std::cout << phi.phi(5924416, c-1) << std::endl;

  std::cout << h_sum << std::endl;

  s3 = thrust::reduce(thrust::device, d_sums, d_sums + y/2);

  cudaFree(d_sums);

  std::cout << num << std::endl;

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
  uint32_t m = 2 * tidx +1;

  if(m <= y && m >= 1 + (y / p)){
    int8_t mu = d_mu[tidx];
    if(d_lpf[tidx] > p){
      uint64_t n = x / (p * m);
      int64_t phi = d_phi[n/2];
      d_sums[tidx] -= (mu * phi);
    }
  }
}
