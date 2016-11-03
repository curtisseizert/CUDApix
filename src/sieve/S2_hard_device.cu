#include <cuda.h>
#include <math_functions.h>
#include <stdint.h>
#include <stdio.h>

#include "sieve/phisieve_constants.cuh"
#include "sieve/S2_hard_device.cuh"
#include "sieve/S2_hard_host.cuh"

__constant__ const uint16_t threadsPerBlock = 256;

__device__
void mRange::update(uint32_t p, const nRange & n)
{
  lo = cdata.x / (n.hi * p);
  lo = lo < cdata.y/p ? cdata.y/p : lo;
  lo = lo > cdata.y ? cdata.y : lo;
  hi = cdata.x / ((n.lo + 1) * p);
  hi = hi < cdata.y/p ? cdata.y/p : hi;
  hi = hi > cdata.y ? cdata.y : hi;
}

__device__
void mRange::update_hi(uint32_t p, const nRange & n)
{
  // uint64_t ub = min(cdata.x / (p * p * p), cdata.z/p);
  uint64_t ub = min(cdata.y, cdata.z/p);

  lo = cdata.x / (n.hi * p);
  lo = lo <= p ? p + 2 : lo;
  lo = lo > ub ? ub : lo;
  hi = cdata.x / ((n.lo + 1) * p);
  hi = hi <= p ? p + 2 : hi;
  hi = hi > ub ? ub : hi;
}

__device__
nRange::nRange(uint64_t bstart)
{
  bitRange = 32 * cdata.sieveWords / threadsPerBlock;
  tfw = threadIdx.x * cdata.sieveWords / threadsPerBlock;
  lo = bstart + 64 * threadIdx.x * (cdata.sieveWords / threadsPerBlock);
  hi = bstart + 64 * (threadIdx.x + 1) * (cdata.sieveWords / threadsPerBlock);
}

__device__
void S2dev::sieveInit(uint32_t * s_sieve, uint64_t bstart)
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
void S2dev::markSmallPrimes(uint32_t * s_sieve, uint64_t bstart, uint16_t a)
{
  for(uint32_t i = threadIdx.x; i < cdata.sieveWords; i += blockDim.x){
    uint64_t j = bstart/64 + i;
    s_sieve[i] |= d_bitmask[a - 1][j % d_smallPrimes[a - 1]];
  }
}

__device__
void S2dev::markMedPrimes(uint32_t * s_sieve, nRange & nr, uint32_t p,
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
void S2dev::markLargePrimes(uint32_t * s_sieve, nRange & nr, uint32_t p,
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
uint32_t S2dev::getCount(uint32_t * s_sieve)
{
  uint32_t c = 0, threadWords = cdata.sieveWords/blockDim.x;
  uint16_t start = threadIdx.x * threadWords;

  for(uint16_t i = start; i < start+threadWords; i++){
    c += __popc(~s_sieve[i]);
  }
  return c;
}

__device__
void S2dev::updatePrimeCache(uint32_t * d_primeList, uint32_t * s_primeCache,
                             uint32_t firstpos, uint32_t primeListLength)
{
  if(threadIdx.x + firstpos < primeListLength)
    s_primeCache[threadIdx.x] = d_primeList[threadIdx.x + firstpos];
}

template <typename T, typename U>
__device__
void S2dev::exclusiveScan(T threadCount, T * s_array, U & total)
{
  s_array[threadIdx.x] = threadCount;
  T sum = 0;
  __syncthreads();

  total = S2dev::inclusiveScan(s_array);

  if(threadIdx.x != 0)
    sum = s_array[threadIdx.x-1];
  else
    sum = 0;

  __syncthreads();
  s_array[threadIdx.x] = sum;
  __syncthreads();
}

template __device__ void S2dev::exclusiveScan<uint32_t, int64_t>(uint32_t, uint32_t *, int64_t &);
template __device__ void S2dev::exclusiveScan<uint32_t, uint64_t>(uint32_t, uint32_t *, uint64_t &);

__device__
uint64_t S2dev::exclusiveScan(int64_t * s_array)
{
  int64_t sum;
  uint64_t total;
  total = S2dev::inclusiveScan(s_array);

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
T S2dev::inclusiveScan(T * s_array)
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

template __device__ int16_t S2dev::inclusiveScan<int16_t> (int16_t * s_array);
template __device__ int32_t S2dev::inclusiveScan<int32_t> (int32_t * s_array);
template __device__ int64_t S2dev::inclusiveScan<int64_t> (int64_t * s_array);

template <typename T>
__device__
void S2dev::zero(T * array, uint32_t numElements)
{
  for(uint32_t i = threadIdx.x; i < numElements; i += blockDim.x)
    array[i] ^= array[i];
}

template __device__ void S2dev::zero<int16_t> (int16_t *, uint32_t);
template __device__ void S2dev::zero<int32_t> (int32_t *, uint32_t);
template __device__ void S2dev::zero<uint32_t> (uint32_t *, uint32_t);


__device__ uint32_t S2dev::countUpSieve(uint32_t * s_sieve, uint16_t firstBit,
                                        uint16_t lastBit, uint16_t threadFirstWord)
{
  uint32_t count = 0;
  for(uint16_t i = firstBit; i < lastBit; i++){
    count += 1u & ~(s_sieve[threadFirstWord + i/32] >> (i & 31));
  }
  return count;
}

__device__
void S2dev::computeMuPhi( uint32_t * s_count, uint32_t * s_sieve, int16_t * s_num,
                          int32_t * s_sums, uint32_t p, S2data_64 * data, nRange & nr)
{
  int32_t muPhi = 0;
  uint32_t phi = s_count[threadIdx.x];
  uint16_t currentBit = 0;
  mRange mr;
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
        phi += S2dev::countUpSieve(s_sieve, currentBit, (1 + n - nr.lo) >> 1, nr.tfw);
        currentBit = (1 + n - nr.lo) >> 1;
        muPhi += phi * mu;
        s_num[threadIdx.x] -= mu;
      }
    }
  }

  s_sums[threadIdx.x] -= muPhi;
}

__device__
void S2dev::computeMuPhiSparse( uint32_t * s_count, uint32_t * s_sieve,
                                int16_t * s_num, int32_t * s_sums, uint32_t p,
                                S2data_64 * data, nRange & nr)
{
  int32_t muPhi = 0;
  uint32_t phi = s_count[threadIdx.x];
  uint16_t currentBit = 0;
  mRange mr;
  mr.update_hi(p, nr);
  s_num[threadIdx.x] = 0;

// as m decreases x/(m * p) increases, so decrementing m allows us to
// count up through the sieve to get the appropriate value of phi

  bool wouldMiss = (cdata.x /( mr.hi * p) <= nr.hi) && (cdata.x /( mr.hi * p) >= nr.lo) && (mr.hi > 1 + cdata.y / p);
  for(uint64_t m = mr.hi - (1 - mr.hi & 1ull); m > mr.lo - wouldMiss; m -= 2){
    uint32_t s = data->d_bitsieve[m/64];
    if((1u & ~(s >> ((m % 64)/2))) == 1u){
      uint64_t n = cdata.x / (m * p);
      // if(p> 3769) printf("%llu\n", m);
      if(n <= nr.hi){
        phi += S2dev::countUpSieve(s_sieve, currentBit, (1 + n - nr.lo) >> 1, nr.tfw);
        currentBit = (1 + n - nr.lo) >> 1;
        muPhi -= phi;
        s_num[threadIdx.x]++;
      }
    }
  }

  s_sums[threadIdx.x] -= muPhi;
}

__global__ void S2glob::S2ctl(S2data_64 * data)
{
  // uint64_t bstart = cdata.bstart + cdata.sieveWords * 64 * blockIdx.x;
  // __shared__ uint32_t s_primeCache[threadsPerBlock]; // this is because many threads
                                                     // will be reading the same prime
                                                     // simultaneously
  __shared__ extern uint32_t s_sieve[];//[cdata.sieveWords];           // where the magic happens
  __shared__ uint32_t s_count[threadsPerBlock];      // stores counts below each thread's
                                                     // group of words in the sieve
  __shared__ int32_t s_sums[threadsPerBlock];        // stores mu(m)*phi(x/(p*m) - b, a)
  __shared__ int16_t s_num[threadsPerBlock];         // stores sum(mu(m)) for each thread

  S2dev::zero(s_sums, threadsPerBlock);
  S2dev::zero(s_count, threadsPerBlock);
  S2dev::zero(s_num, threadsPerBlock);
  __syncthreads();

  nRange nr(cdata.bstart + cdata.sieveWords * 64 * blockIdx.x);

  uint32_t pi_p = cdata.c - 1;
  uint32_t p = d_smallPrimes[pi_p];

  S2dev::sieveInit(s_sieve, cdata.bstart + cdata.sieveWords * 64 * blockIdx.x);
  __syncthreads();
  uint32_t threadCount = S2dev::getCount(s_sieve);
  __syncthreads();
  S2dev::exclusiveScan(threadCount, s_count, data->d_sums[pi_p * cdata.blocks + blockIdx.x]);
  __syncthreads();
  // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\n", p);
  S2dev::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
  __syncthreads();

  data->d_num[pi_p * cdata.blocks + blockIdx.x] = S2dev::inclusiveScan(s_num);
  __syncthreads();

  while(pi_p < cutoff){
    pi_p++;
    S2dev::markSmallPrimes(s_sieve, cdata.bstart + cdata.sieveWords * 64 * blockIdx.x, pi_p);
    __syncthreads();
    threadCount = S2dev::getCount(s_sieve);
     __syncthreads();
    S2dev::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();
    p = d_smallPrimes[pi_p];
    // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\n", p);
    S2dev::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = S2dev::inclusiveScan(s_num);
    __syncthreads();
  }

  while(pi_p < cdata.primeListLength - 3){

    pi_p++;
    S2dev::markMedPrimes(s_sieve, nr, p, threadCount);
    __syncthreads();

    S2dev::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();

    p = data->d_primeList[pi_p + 1];
    if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\n", p);
    if(p > __float2uint_rd(sqrtf(cdata.y))) goto Sparse;

    S2dev::computeMuPhi(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = S2dev::inclusiveScan(s_num);
    __syncthreads();
  }

  while(pi_p < cdata.primeListLength - 3){

    pi_p++;
    S2dev::markMedPrimes(s_sieve, nr, p, threadCount);
    __syncthreads();

    S2dev::exclusiveScan(threadCount, s_count, data->d_sums[(pi_p) * cdata.blocks + blockIdx.x]);
    __syncthreads();

    p = data->d_primeList[pi_p + 1];
    // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\n", p);
Sparse:
    if(cdata.x/(p * p) < cdata.bstart + cdata.sieveWords * 64 * blockIdx.x) goto End;
    S2dev::computeMuPhiSparse(s_count, s_sieve, s_num, s_sums, p, data, nr);
    __syncthreads();
    data->d_num[pi_p * cdata.blocks + blockIdx.x] = S2dev::inclusiveScan(s_num);
    __syncthreads();
  }
End:
  __syncthreads();
  data->d_partialsums[blockIdx.x] = S2dev::inclusiveScan(s_sums);
 } // phew!

__global__ void S2glob::scanVectorized(int64_t * a, uint64_t * overflow)
{
  __shared__ extern int64_t s_a[];
  s_a[threadIdx.x] = a[blockIdx.x*blockDim.x+threadIdx.x];
  __syncthreads();

  overflow[blockIdx.x] += S2dev::exclusiveScan(s_a);
  __syncthreads();

  a[blockIdx.x*blockDim.x+threadIdx.x] = s_a[threadIdx.x];
}

__global__ void S2glob::addMultiply(int64_t * a, uint64_t * b, int16_t * c) // a = (a + b) * c
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  int64_t d = (int64_t)b[blockIdx.x];
  d += a[tidx];
  a[tidx] = d * c[tidx];
}

__global__ void S2glob::addArrays(uint64_t * a, uint64_t * b, uint32_t numElements) // a += b
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  if(tidx < numElements)
    a[tidx] += b[tidx];
}

__global__ void testRed(int32_t * array)
{
  __shared__ int32_t s_array[256];
  s_array[threadIdx.x] = array[threadIdx.x];
  __syncthreads();

  S2dev::inclusiveScan(s_array);
  __syncthreads();

  array[threadIdx.x] = s_array[threadIdx.x];
}

__host__ void S2hardHost::transferConstants()
{
  cudaMemcpyToSymbolAsync(cdata, &h_cdata, sizeof(c_data64), 0, cudaMemcpyDefault);
}
