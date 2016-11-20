#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <CUDASieve/launch.cuh>
#include <CUDASieve/primelist.cuh>
#include <CUDASieve/cudasieve.hpp>

#include "sieve/lpf_mu.cuh"

/*

**To do**
- make a version of lpf that only goes up to the nth prime
- fix 64 bit mobius function

 */

const uint16_t h_sieveBytes = 16384;
__constant__ uint16_t sieveBytes = 16384;
__constant__ uint8_t numSmallPrimes = 11;
__constant__ uint32_t smallPrimes[11] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

__global__ void lpf_kernel(uint32_t * d_primeList, uint32_t * d_lpf, uint32_t primeListLength, uint16_t sieveWords, uint32_t bottom)
{
  uint32_t bstart = 2 * blockIdx.x * sieveWords + bottom;
  __shared__ extern uint32_t s_lpf32[];

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
     s_lpf32[i] = 2 * i + bstart + 1;
     for(uint16_t j = 0; j < numSmallPrimes; j++){
       if((2*i+1 + bstart) % smallPrimes[j] == 0) atomicMin(&s_lpf32[i], smallPrimes[j]);
     }
  }

  __syncthreads();

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p = d_primeList[i];
      uint32_t off = p - bstart % p;
      if(off%2==0) off += p;
      off = off >> 1;
      for(; off < sieveWords; off += p) atomicMin(&s_lpf32[off], p);
  }

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x)
      d_lpf[i + sieveWords*blockIdx.x] = s_lpf32[i];

  __syncthreads();

  if(threadIdx.x == 0 && blockIdx.x == 0) d_lpf[0] = (uint32_t) -1;
}

__global__ void lpf_kernel(uint32_t * d_primeList, uint64_t * d_lpf, uint32_t primeListLength, uint16_t sieveWords, uint64_t bottom)
{
  uint64_t bstart = 2 * blockIdx.x * sieveWords + bottom;
  __shared__ extern uint64_t s_lpf64[];

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
     s_lpf64[i] = 2 * i + bstart + 1;
     for(uint16_t j = 0; j < numSmallPrimes; j++){
       if((2*i+1 + bstart) % smallPrimes[j] == 0) atomicMin((unsigned long long *)&s_lpf64[i], (unsigned long long)smallPrimes[j]);
     }
  }

  __syncthreads();

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p = d_primeList[i];
      uint32_t off = p - bstart % p;
      if(off%2==0) off += p;
      off = off >> 1;
      for(; off < sieveWords; off += p) atomicMin((unsigned long long *)&s_lpf64[off], (unsigned long long)p);
  }

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x)
      d_lpf[i + sieveWords*blockIdx.x] = s_lpf64[i];

  __syncthreads();

  if(threadIdx.x == 0 && blockIdx.x == 0) d_lpf[0] = (uint64_t) -1;
}

__device__ void atomicMult(int32_t * addr, int32_t val)
{
  int old = *addr, assumed;

  do{
    assumed = old;
    old = atomicCAS((unsigned int*) addr, (unsigned int) assumed,(unsigned int) val * assumed);
  }while(assumed != old);
}

__device__ void atomicMult(int64_t * addr, int64_t val)
{
  int old = *addr, assumed;

  do{
    assumed = old;
    old = atomicCAS((long long unsigned int*) addr, (long long unsigned int) assumed,(long long unsigned int) val * assumed);
  }while(assumed != old);
}

__global__ void mu_kernel(uint32_t * d_primeList, int8_t * d_mu,
                          uint32_t primeListLength, uint16_t sieveWords,
                          uint32_t bottom)
{
  uint32_t bstart = 2 * blockIdx.x * sieveWords + bottom;
  __shared__ extern int32_t s_mu32[];

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
     s_mu32[i] = 1;
     for(uint16_t j = 0; j < numSmallPrimes; j++){
       if((2*i+1 + bstart) % smallPrimes[j] == 0) s_mu32[i] *= -smallPrimes[j];
       if((2*i+1 + bstart) % (smallPrimes[j] * smallPrimes[j]) == 0) s_mu32[i] = 0;
     }
  }

  __syncthreads();

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p = d_primeList[i];
      uint32_t off = p - bstart % p;
      if(off%2==0) off += p;
      off = off >> 1;
      for(; off < sieveWords; off += p) atomicMult(&s_mu32[off], -p);
  }

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p_squared = d_primeList[i] * d_primeList[i];
      uint32_t off = p_squared - bstart % p_squared;
      if(off%2==0) off += p_squared;
      off = off >> 1;
      for(; off < sieveWords; off += p_squared) atomicExch(&s_mu32[off], (int) 0);
  }

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
    if(abs(s_mu32[i]) != 2 * i + bstart + 1) s_mu32[i] *= -1;
    if(s_mu32[i] == 0)          d_mu[i + sieveWords*blockIdx.x] = 0;
    else if(s_mu32[i] < 0)      d_mu[i + sieveWords*blockIdx.x] = -1;
    else if(s_mu32[i] > 0)      d_mu[i + sieveWords*blockIdx.x] = 1;
  }
}

// 64 bit mu - not working!!
__global__ void mu_kernel(uint32_t * d_primeList, int8_t * d_mu,
                          uint32_t primeListLength, uint16_t sieveWords,
                          uint64_t bottom)
{
  uint64_t bstart = 2 * blockIdx.x * sieveWords + bottom;
  __shared__ extern int64_t s_mu64[];

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
     s_mu64[i] = 1;
     for(uint16_t j = 0; j < numSmallPrimes; j++){
       if((2*i+1 + bstart) % smallPrimes[j] == 0) s_mu64[i] *= -smallPrimes[j];
       if((2*i+1 + bstart) % (smallPrimes[j] * smallPrimes[j]) == 0) s_mu64[i] = 0;
     }
  }

  __syncthreads();

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p = d_primeList[i];
      uint32_t off = p - bstart % p;
      if(off%2==0) off += p;
      off = off >> 1;
      for(; off < sieveWords; off += p) atomicMult(&s_mu64[off], (int64_t)-p);
  }

  for(uint32_t i = threadIdx.x; i < primeListLength; i+= blockDim.x){
      uint32_t p_squared = d_primeList[i] * d_primeList[i];
      uint32_t off = p_squared - bstart % p_squared;
      if(off%2==0) off += p_squared;
      off = off >> 1;
      for(; off < sieveWords; off += p_squared) atomicExch((long long unsigned*)&s_mu64[off], 0ull);
  }

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x){
    if(abs(s_mu64[i]) != (uint32_t)(2 * i + bstart + 1)) s_mu64[i] *= -1;
    if(s_mu64[i] == 0)          d_mu[i + sieveWords*blockIdx.x] = 0;
    else if(s_mu64[i] < 0)      d_mu[i + sieveWords*blockIdx.x] = -1;
    else if(s_mu64[i] > 0)      d_mu[i + sieveWords*blockIdx.x] = 1;
  }
}

template <typename T>
T* gen_d_lpf(T bottom, T top)
{
  uint16_t threads = 256, sieveWords = h_sieveBytes/(sizeof(T));
  uint32_t * d_primeList, primeListLength, blocks = 1 + (top - bottom) / (2 * sieveWords);
  uint64_t arraySize = ((top - bottom) / 2) + sieveWords - (((top - bottom) / 2) % sieveWords);
  T * d_lpf = NULL;

  d_lpf = safeCudaMalloc(d_lpf, arraySize * sizeof(T));

  d_primeList = PrimeList::getSievingPrimes((uint32_t) sqrt(top), primeListLength, 1);

  lpf_kernel<<<blocks, threads, h_sieveBytes>>>(d_primeList, d_lpf, primeListLength, sieveWords, bottom);

  cudaDeviceSynchronize();
  cudaFree(d_primeList);

  return d_lpf;
}

template uint32_t * gen_d_lpf<uint32_t>(uint32_t bottom, uint32_t top);
template uint64_t * gen_d_lpf<uint64_t>(uint64_t bottom, uint64_t top);

template <typename T>
T * gen_h_lpf(T bottom, T top)
{
  T * h_lpf, * d_lpf;
  T arraySize = (top - bottom)/2 + 1;

  d_lpf = gen_d_lpf(bottom, top);

  h_lpf = (T *)malloc(arraySize * sizeof(T));
  cudaMemcpy(h_lpf, d_lpf, arraySize * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_lpf);

  return h_lpf;
}

template uint32_t * gen_h_lpf<uint32_t>(uint32_t bottom, uint32_t top);
template uint64_t * gen_h_lpf<uint64_t>(uint64_t bottom, uint64_t top);

template <typename T>
int8_t * gen_d_mu(T bottom, T top)
{
  uint16_t threads = 256, sieveWords = h_sieveBytes/(sizeof(T));
  uint32_t * d_primeList, primeListLength, blocks = 1 + (top - bottom) / (2 * sieveWords);
  uint64_t arraySize = ((top - bottom) / 2) + sieveWords - (((top - bottom) / 2) % sieveWords);
  int8_t * d_mu = NULL;

  d_mu = safeCudaMalloc(d_mu, arraySize * sizeof(int8_t));

  d_primeList = PrimeList::getSievingPrimes((uint32_t) sqrt(top), primeListLength, 1);

  mu_kernel<<<blocks, threads, h_sieveBytes>>>(d_primeList, d_mu, primeListLength, sieveWords, bottom);

  cudaDeviceSynchronize();

  return d_mu;
}

template int8_t * gen_d_mu<uint32_t>(uint32_t bottom, uint32_t top);
template int8_t * gen_d_mu<uint64_t>(uint64_t bottom, uint64_t top);

template <typename T>
int8_t * gen_h_mu(T bottom, T top)
{
  int8_t * h_mu, * d_mu;
  T arraySize = (top - bottom)/2 + 1;

  d_mu = gen_d_mu(bottom, top);

  h_mu = (int8_t * )malloc(arraySize * sizeof(int8_t));
  cudaMemcpy(h_mu, d_mu, arraySize * sizeof(int8_t), cudaMemcpyDeviceToHost);

  cudaFree(d_mu);

  return h_mu;
}

template int8_t * gen_h_mu<uint32_t>(uint32_t bottom, uint32_t top);
template int8_t * gen_h_mu<uint64_t>(uint64_t bottom, uint64_t top);
