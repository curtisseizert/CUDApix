#include <stdint.h>
#include <cuda.h>
#include <cuda_uint128.h>

#ifndef _OMEGA12
#define _OMEGA12

struct cdata128{
  uint128_t x;
  uint64_t bstart, mstart, sqrty, y, z;
  uint32_t blocks, elPerBlock, maxPrime;
  size_t primeListLength;
  uint16_t c, sieveWords;
};

__constant__ cdata128 cdata;

struct omega12data_128{
  int64_t * d_sums, * d_partialsums;
  uint64_t * d_totals, * d_lpf, * d_totalsNext;
  uint32_t * d_primeList, * d_bitsieve;
  int16_t * d_num;
  int8_t * d_mu;
};

struct nRange128{
  uint64_t lo, hi;
  uint16_t tfw, bitRange; // tfw = threadFirstWord
  __device__ inline nRange128(uint64_t bstart);
};

struct mRange128{
  uint64_t lo, hi;
  __device__ void update(uint32_t p, const nRange128 & n);
  __device__ void update_hi(uint32_t p, const nRange128 & n);
};


namespace omega12{
  __device__ void sieveInit(uint32_t * d_sieve, uint64_t bstart);
  __device__ void markSmallPrimes(uint32_t * s_sieve, uint64_t bstart, uint16_t c);
  __device__ void markMedPrimes(uint32_t * s_sieve, nRange128 & nr, uint32_t p, uint32_t & threadCount);
  __device__ void markLargePrimes(uint32_t * s_sieve, nRange128 & nr, uint32_t p, uint32_t & threadCount);
  __device__ uint32_t getCount(uint32_t * s_sieve);
  __device__ void updatePrimeCache(uint32_t * d_primeList, uint32_t * s_primeCache, uint32_t first, uint32_t primeListLength);
  __device__ uint64_t exclusiveScan(int64_t * array);
  __device__ void computeMuPhi(uint32_t * s_count, uint32_t * s_sieve, int16_t * s_num, int32_t * s_sums, uint32_t p, omega12data_128 * data, nRange128 & nr); // want to -= this quantity
  __device__ void computeMuPhiSparse(uint32_t * s_count, uint32_t * s_sieve, int16_t * s_num, int32_t * s_sums, uint32_t p, omega12data_128 * data, nRange128 & nr); // want to -= this quantity
  __device__ uint32_t countUpSieve(uint32_t * s_sieve, uint16_t firstBit, uint16_t lastBit, uint16_t threadFirstWord);

  template <typename T>
  __device__ void zero(T * array, uint32_t numElements);

  template<typename T>
  __device__ T inclusiveScan(T * s_array);

  template <typename T, typename U>
  __device__ void exclusiveScan(T threadCount, T * s_array, U & total);

}

namespace Omega12Global{
  __global__ void omega12_ctl(omega12data_128 * data);
  __global__ void scanVectorized(int64_t * a, uint64_t * overflow);
  __global__ void addMultiply(int64_t * a, uint64_t * b, int16_t * c); // a = (a + b) * c
  __global__ void multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements);
  __global__ void addArrays(uint64_t * a, uint64_t * b, uint32_t numElements); // a += b
}

__global__ void testRed(int32_t * array);

class Omega12Host{
private:
  omega12data_128 * data;
  cdata128 h_cdata;
  uint16_t numStreams_ = 5, sieveWords_ = 3072;
  cudaStream_t stream[5];
  uint64_t maxPrime_;

public:

  Omega12Host(uint128_t x, uint64_t y, uint16_t c, uint64_t maxPrime);
  Omega12Host(uint128_t x, uint64_t y, uint16_t c);
  ~Omega12Host();

  void makeData(uint128_t x, uint64_t y, uint16_t c);

  void setupNextIter();

  void allocate();
  void deallocate();
  void zero();
  void transferConstants();

  uint128_t launchIter();


  static uint128_t omega12(uint128_t x, uint64_t y, uint16_t c);
};

#endif
