#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <CUDASieve/cudasieve.hpp>

#include "sieve/lpf_mu.cuh"
#include "phi.cuh"

//const uint16_t pi_lookup[20] = {0,0,1,2,2,3,3,4,4,4,4,5,5,6,6,6,6,7,7,8}; //starts at 0
const uint16_t smallPrimes[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

Phi::Phi(uint32_t max)
{
  d_lpf = gen_d_lpf(0u, max);
  cudaDeviceSynchronize();
  max_ = max;

  d_primeList = PrimeList::getSievingPrimes((uint32_t) sqrt(max), primeListLength, 1);
  cudaMallocHost(&h_primeList, primeListLength * sizeof(uint32_t));
  cudaMemcpy(h_primeList, d_primeList, primeListLength * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

Phi::Phi(uint32_t max, uint32_t p_max)
{
  d_lpf = gen_d_lpf(0u, max);
  cudaDeviceSynchronize();
  max_ = max;
  p_max_ = p_max;

  d_primeList = PrimeList::getSievingPrimes(p_max, primeListLength, 1);
  cudaMallocHost(&h_primeList, primeListLength * sizeof(uint32_t));
  cudaMemcpy(h_primeList, d_primeList, primeListLength * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

Phi::~Phi()
{
  if(d_lpf != NULL) cudaFree(d_lpf);
}

uint64_t Phi::phi(uint32_t x, uint32_t a)
{
  uint32_t y;
  if(a > 12) y = h_primeList[a - 12];
  else y = smallPrimes[a];

  uint64_t count, x_corr = (x - (x & 1ull))/2;

  greater_than_n op_(y);

  count = 1 + thrust::count_if(thrust::device, d_lpf + y/2, d_lpf + x_corr, op_);

  return count;
}

void Phi::makeModCache(uint32_t pi_max)
{
  c_ = pi_max;
  uint32_t wheelMod = 6;

  h_cache = (uint32_t *)malloc(c_ * sizeof(uint32_t));
  h_cache[0] = 1;
  h_cache[1] = 2;

  for(uint32_t i = 3; i < c_; i++){
    wheelMod *= smallPrimes[i];
    h_cache[i-1] = phi(wheelMod, i);
    std::cout << wheelMod << "\t\t" << h_cache[i -1] << std::endl;
    uint32_t * d_phi = generateRange(wheelMod, smallPrimes[i]);
    cudaFree(d_phi);
  }
}

uint32_t * Phi::generateRange(uint32_t num, uint32_t a)
{
  num /= 2;
  uint32_t * d_phi;
  uint16_t threads = 256;
  uint32_t blocks = 1 + num/(threads), y;
  if(a > 12) y = h_primeList[a - 12];
  else y = smallPrimes[a];

  cudaMalloc(&d_phi, num * sizeof(uint32_t));

  isGreaterThan<<<blocks, threads>>>(num, d_lpf, d_phi, y);
  //addOneToStart<<<1,1>>>(d_phi);

  cudaDeviceSynchronize();

  thrust::inclusive_scan(thrust::device, d_phi, d_phi + num, d_phi);

  cudaDeviceSynchronize();

  return d_phi;
}

__global__ void addOneToStart(uint32_t * a)
{
  if(threadIdx.x == 0) a[0]++;
}

__global__ void isGreaterThan(uint32_t num, uint32_t * d_arrayIn, uint32_t * d_arrayOut, uint32_t value)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tidx < num)
    d_arrayOut[tidx] = (d_arrayIn[tidx] > value) ? 1 : 0;
}
