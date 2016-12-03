#include <cuda.h>
#include <math_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_uint128.h>

#include "sieve/phisieve_constants.cuh"
#include "Deleglise-Rivat/omega12.cuh"

__constant__ const uint16_t threadsPerBlock = 256;

__device__
void mRange128::update(uint32_t p, const nRange128 & n)
{
  lo = cdata.x / (n.hi * p);
  lo = lo < cdata.y/p ? cdata.y/p : lo;
  lo = lo > cdata.y ? cdata.y : lo;
  hi = cdata.x / ((n.lo + 1) * p);
  hi = hi < cdata.y/p ? cdata.y/p : hi;
  hi = hi > cdata.y ? cdata.y : hi;
}

__device__
void mRange128::update_hi(uint32_t p, const nRange128 & n)
{
  uint64_t ub = min(div128to64(cdata.x, (p * p * p)), cdata.y);

  lo = cdata.x / (n.hi * p);
  lo = lo <= p ? p + 2 : lo;
  lo = lo > ub ? ub : lo;
  hi = cdata.x / ((n.lo + 1) * p);
  hi = hi <= p ? p + 2 : hi;
  hi = hi > ub ? ub : hi;
}

__device__
nRange128::nRange128(uint64_t bstart)
{
  bitRange = 32 * cdata.sieveWords / threadsPerBlock;
  tfw = threadIdx.x * cdata.sieveWords / threadsPerBlock;
  lo = bstart + 64 * threadIdx.x * (cdata.sieveWords / threadsPerBlock);
  hi = bstart + 64 * (threadIdx.x + 1) * (cdata.sieveWords / threadsPerBlock);
}

__device__
void omega12::sieveInit(uint32_t * s_sieve, uint64_t bstart)
{
  for(uint32_t i = threadIdx.x; i < cdata.sieveWords; i += blockDim.x){
    uint64_t j = bstart/64 + i;
    uint32_t s = d_bitmask[0][j % d_smallPrimes[0]];
    for(uint16_t a = 1; a < cdata.c - 1; a++)
      s |= d_bitmask[a][j % d_smallPrimes[a]];
    s_sieve[i] = s;
  }
}

__device__
void omega12::markSmallPrimes(uint32_t * s_sieve, uint64_t bstart, uint16_t a)
{
  for(uint32_t i = threadIdx.x; i < cdata.sieveWords; i += blockDim.x){
    uint64_t j = bstart/64 + i;
    s_sieve[i] |= d_bitmask[a - 1][j % d_smallPrimes[a - 1]];
  }
}

__device__
void omega12::markMedPrimes(uint32_t * s_sieve, nRange128 & nr, uint32_t p,
                          uint32_t & threadCount)
{
  uint32_t off = p - nr.lo % p;
  if(off%2==0) off += p;
  off = off >> 1; // convert offset to align with half sieve
  for(; off < nr.bitRange; off += p){
    threadCount -= (1u & ~(s_sieve[nr.tfw + (off >> 5)] >> (off & 31)));
    s_sieve[nr.tfw + (off >> 5)] |= (1u << (off & 31));
  }
}

__device__
void omega12::markLargePrimes(uint32_t * s_sieve, nRange128 & nr, uint32_t p,
                            uint32_t & threadCount)
{
  uint32_t off = p - nr.lo % p;
  if(off%2==0) off += p;
  off = off >> 1; // convert offset to align with half sieve
  if(off < nr.bitRange){
    threadCount -= ~(s_sieve[nr.tfw + (off >> 5)] >> (off & 31));
    s_sieve[nr.tfw + (off >> 5)] |= (1u << (off & 31));
  }
}

__device__
uint32_t omega12::getCount(uint32_t * s_sieve)
{
  uint32_t c = 0, threadWords = cdata.sieveWords/blockDim.x;
  uint16_t start = threadIdx.x * threadWords;

  for(uint16_t i = start; i < start+threadWords; i++){
    c += __popc(~s_sieve[i]);
  }
  return c;
}

__device__
void omega12::updatePrimeCache(uint32_t * d_primeList, uint32_t * s_primeCache,
                             uint32_t firstpos, uint32_t primeListLength)
{
  if(threadIdx.x + firstpos < primeListLength)
    s_primeCache[threadIdx.x] = d_primeList[threadIdx.x + firstpos];
}

template <typename T, typename U>
__device__
void omega12::exclusiveScan(T threadCount, T * s_array, U & total)
{
  s_array[threadIdx.x] = threadCount;
  T sum = 0;
  __syncthreads();

  total = omega12::inclusiveScan(s_array);

  if(threadIdx.x != 0)
    sum = s_array[threadIdx.x-1];
  else
    sum = 0;

  __syncthreads();
  s_array[threadIdx.x] = sum;
  __syncthreads();
}

template __device__ void omega12::exclusiveScan<uint32_t, int64_t>(uint32_t, uint32_t *, int64_t &);
template __device__ void omega12::exclusiveScan<uint32_t, uint64_t>(uint32_t, uint32_t *, uint64_t &);

__device__
uint64_t omega12::exclusiveScan(int64_t * s_array)
{
  int64_t sum;
  uint64_t total;
  total = omega12::inclusiveScan(s_array);

  if(threadIdx.x != 0)
    sum = s_array[threadIdx.x-1];
  else
    sum = 0;
   __syncthreads();

  s_array[threadIdx.x] = sum;
  return total;
}

template<typename T>
__device__
T omega12::inclusiveScan(T * s_array)
{
  T sum;

  for(uint32_t offset = 1; offset <= threadsPerBlock; offset *= 2){
    if(threadIdx.x >= offset){
      sum = s_array[threadIdx.x] + s_array[threadIdx.x - offset];
    }else{sum = s_array[threadIdx.x];}
    __syncthreads();
    s_array[threadIdx.x] = sum;
    __syncthreads();
  }

  __syncthreads();
  s_array[threadIdx.x] = sum;
  __syncthreads();
  return s_array[blockDim.x-1];
}

template __device__ int16_t omega12::inclusiveScan<int16_t> (int16_t * s_array);
template __device__ int32_t omega12::inclusiveScan<int32_t> (int32_t * s_array);
template __device__ int64_t omega12::inclusiveScan<int64_t> (int64_t * s_array);

template <typename T>
__device__
void omega12::zero(T * array, uint32_t numElements)
{
  for(uint32_t i = threadIdx.x; i < numElements; i += blockDim.x)
    array[i] ^= array[i];
}

template __device__ void omega12::zero<int16_t> (int16_t *, uint32_t);
template __device__ void omega12::zero<int32_t> (int32_t *, uint32_t);
template __device__ void omega12::zero<uint32_t> (uint32_t *, uint32_t);


__device__ uint32_t omega12::countUpSieve(uint32_t * s_sieve, uint16_t firstBit,
                                        uint16_t lastBit, uint16_t threadFirstWord)
{
  uint32_t count = 0;
  for(uint16_t i = firstBit; i < lastBit; i++){
    count += 1u & ~(s_sieve[threadFirstWord + i/32] >> (i & 31));
  }
  return count;
}

__device__
void omega12::computeMuPhi( uint32_t * s_count, uint32_t * s_sieve, int16_t * s_num,
                          int32_t * s_sums, uint32_t p, omega12data_128 * data, nRange128 & nr)
{
  int32_t muPhi = 0;
  uint32_t phi = s_count[threadIdx.x];
  uint16_t currentBit = 0;
  mRange128 mr;
  mr.update(p, nr);
  s_num[threadIdx.x] = 0;

// as m decreases x/(m * p) increases, so decrementing m allows us to
// count up through the sieve to get the appropriate value of phi

  bool wouldMiss = (cdata.x /( mr.hi * p) <= nr.hi) && (cdata.x /( mr.hi * p) >= nr.lo) && (mr.hi > 1 + cdata.y / p);
  for(uint64_t m = mr.hi - (1 - mr.hi & 1ull); m > mr.lo - wouldMiss; m -= 2){
    if(data->d_lpf[((m - cdata.mstart) >> 1)] > p){
      int8_t mu = data->d_mu[(m - cdata.mstart) >> 1];
      uint64_t n = cdata.x / (m * p);
      if(mu != 0 && n <= nr.hi){
        phi += omega12::countUpSieve(s_sieve, currentBit, (1 + n - nr.lo) >> 1, nr.tfw);
        currentBit = (1 + n - nr.lo) >> 1;
        muPhi += phi * mu;
        s_num[threadIdx.x] -= mu;
      }
    }
  }
  s_sums[threadIdx.x] -= muPhi;
}

__device__
void omega12::computeMuPhiSparse( uint32_t * s_count, uint32_t * s_sieve,
                                int16_t * s_num, int32_t * s_sums, uint32_t p,
                                omega12data_128 * data, nRange128 & nr)
{
  int32_t muPhi = 0;
  uint32_t phi = s_count[threadIdx.x];
  uint16_t currentBit = 0;
  mRange128 mr;
  mr.update_hi(p, nr);
  s_num[threadIdx.x] = 0;

// as m decreases x/(m * p) increases, so decrementing m allows us to
// count up through the sieve to get the appropriate value of phi

  bool wouldMiss = (cdata.x /( mr.hi * p) <= nr.hi) && (cdata.x /( mr.hi * p) >= nr.lo) && (mr.hi > 1 + cdata.y / p);
  for(uint64_t m = mr.hi - (1 - mr.hi & 1ull); m > mr.lo - wouldMiss; m -= 2){
    uint32_t s = data->d_bitsieve[m/64];
    if((1u & ~(s >> ((m % 64)/2))) == 1u){
      uint64_t n = cdata.x / (m * p);
      if(n <= nr.hi){
        phi += omega12::countUpSieve(s_sieve, currentBit, (1 + n - nr.lo) >> 1, nr.tfw);
        currentBit = (1 + n - nr.lo) >> 1;
        muPhi -= phi;
        s_num[threadIdx.x]++;
      }
    }
  }

  s_sums[threadIdx.x] -= muPhi;
}

__global__ void Omega12Global::omega12_ctl(omega12data_128 * data)
{
  // uint64_t bstart = cdata.bstart + cdata.sieveWords * 64 * blockIdx.x;
  // __shared__ uint32_t s_primeCache[threadsPerBlock]; // this is because many threads
                                                     // will be reading the same prime
                                                     // simultaneously
  __shared__ extern uint32_t s_sieve[];		           // where the magic happens
  __shared__ uint32_t s_count[threadsPerBlock];      // stores counts below each thread's
                                                     // group of words in the sieve
  __shared__ int32_t s_sums[threadsPerBlock];        // stores mu(m)*phi(x/(p*m) - b, a)
  __shared__ int16_t s_num[threadsPerBlock];         // stores sum(mu(m)) for each thread

  omega12::zero(s_sums, threadsPerBlock);
  omega12::zero(s_count, threadsPerBlock);
  omega12::zero(s_num, threadsPerBlock);
  __syncthreads();

  nRange128 nr(cdata.bstart + cdata.sieveWords * 64 * blockIdx.x);

  uint32_t pi_p = cdata.c - 1;
  uint32_t p = d_smallPrimes[pi_p];

  omega12::sieveInit(s_sieve, cdata.bstart + cdata.sieveWords * 64 * blockIdx.x);
  __syncthreads();
  uint32_t threadCount = omega12::getCount(s_sieve);
  __syncthreads();
  omega12::exclusiveScan(threadCount, s_count, data->d_sums[pi_p * cdata.blocks + blockIdx.x]);
  __syncthreads();
  omega12::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
  __syncthreads();

  data->d_num[pi_p * cdata.blocks + blockIdx.x] = omega12::inclusiveScan(s_num);
  __syncthreads();

  while(pi_p < cutoff){
    pi_p++;
    omega12::markSmallPrimes(s_sieve, cdata.bstart + cdata.sieveWords * 64 * blockIdx.x, pi_p);
    __syncthreads();
    threadCount = omega12::getCount(s_sieve);
     __syncthreads();
    omega12::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();
    p = d_smallPrimes[pi_p];
    omega12::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = omega12::inclusiveScan(s_num);
    __syncthreads();
  }

  while(pi_p < cdata.primeListLength - 3){

    pi_p++;
    omega12::markMedPrimes(s_sieve, nr, p, threadCount);
    __syncthreads();

    omega12::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();

    p = data->d_primeList[pi_p + 1];
    if(p > cdata.sqrty) goto End;

    omega12::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = omega12::inclusiveScan(s_num);
    __syncthreads();
  }

  while(pi_p < cdata.primeListLength - 3){

    pi_p++;
    omega12::markMedPrimes(s_sieve, nr, p, threadCount);
    __syncthreads();

    omega12::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();

    p = data->d_primeList[pi_p + 1];
Sparse:
    if(cdata.x/(p * p * p) < cdata.bstart + cdata.sieveWords * 64 * blockIdx.x) goto End;
    omega12::computeMuPhiSparse(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = omega12::inclusiveScan(s_num);
    __syncthreads();
  }
End:
  __syncthreads();
  data->d_partialsums[blockIdx.x] = omega12::inclusiveScan(s_sums);
 } // phew!

__global__ void Omega12Global::scanVectorized(int64_t * a, uint64_t * overflow)
{
  __shared__ extern int64_t s_a[];
  s_a[threadIdx.x] = a[blockIdx.x*blockDim.x+threadIdx.x];
  __syncthreads();

  overflow[blockIdx.x] += omega12::exclusiveScan(s_a);
  __syncthreads();

  a[blockIdx.x*blockDim.x+threadIdx.x] = s_a[threadIdx.x];
}

__global__ void Omega12Global::addMultiply(int64_t * a, uint64_t * b, int16_t * c) // a = (a + b) * c
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  int64_t d = (int64_t)b[blockIdx.x];
  d += a[tidx];
  a[tidx] = d * c[tidx];
}

__global__ void Omega12Global::addArrays(uint64_t * a, uint64_t * b, uint32_t numElements) // a += b
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  if(tidx < numElements)
    a[tidx] += b[tidx];
}

__host__ void Omega12Host::transferConstants()
{
  cudaMemcpyToSymbolAsync(cdata, &h_cdata, sizeof(cdata128), 0, cudaMemcpyDefault);
}
