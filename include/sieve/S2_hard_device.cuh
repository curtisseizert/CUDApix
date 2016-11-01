#include <stdint.h>
#include <cuda.h>

#ifndef _S2_HARD_DEVICE
#define _S2_HARD_DEVICE

struct S2data_64{
  int64_t * d_sums, * d_partialsums;
  uint64_t bstart, mstart, x, y;
  uint64_t * d_totals, * d_totalsNext;
  uint32_t * d_primeList, * d_lpf;
  uint32_t primeListLength, blocks, elPerBlock;
  uint16_t c;
  int16_t * d_num;
  int8_t * d_mu;
};

struct nRange{
  uint64_t lo, hi;
  uint16_t tfw, bitRange; // tfw = threadFirstWord
  __device__ nRange(uint64_t bstart);
};

struct mRange{
  uint64_t lo, hi;
  __device__ void update(uint64_t x, uint64_t y, uint32_t p, nRange n);
};


namespace S2dev{
  __device__ void sieveInit(uint32_t * d_sieve, uint64_t bstart, uint16_t c);
  __device__ void markSmallPrimes(uint32_t * s_sieve, uint64_t bstart, uint16_t c);
  __device__ void markMedPrimes(uint32_t * s_sieve, nRange & nr, uint32_t p, uint32_t & threadCount);
  __device__ void markLargePrimes(uint32_t * s_sieve, nRange & nr, uint32_t p, uint32_t & threadCount);
  __device__ uint32_t getCount(uint32_t * s_sieve);
  __device__ void updatePrimeCache(uint32_t * d_primeList, uint32_t * s_primeCache, uint32_t first, uint32_t primeListLength);
  __device__ void exclusiveScan(int64_t * array, uint64_t & total);
  __device__ void computeMuPhi(uint32_t * s_count, uint32_t * s_sieve, int16_t * s_num, int32_t * s_sums, uint32_t p, S2data_64 * data, mRange & mr, nRange & nr); // want to -= this quantity
  __device__ uint32_t countUpSieve(uint32_t * s_sieve, uint16_t firstBit, uint16_t lastBit, uint16_t threadFirstWord);

  template <typename T>
  __device__ void zero(T * array, uint32_t numElements);

  template<typename T>
  __device__ T inclusiveScan(T * s_array);

  template <typename T, typename U>
  __device__ void exclusiveScan(T threadCount, T * s_array, U & total);

}

namespace S2glob{
  __global__ void S2ctl(S2data_64 * data);
  __global__ void scanVectorized(int64_t * a, uint64_t * overflow);
  __global__ void addMultiply(int64_t * a, uint64_t * b, int16_t * c); // a = (a + b) * c
  __global__ void multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements);
}

__global__ void testRed(int32_t * array);

#endif
