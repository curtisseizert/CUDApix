#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef _CUDAPIX
#define _CUDAPIX

struct PrimeArray{
  uint64_t * h_primes, * d_primes, bottom, top, pi_bottom, pi_top, diff_pi;
  size_t len;

  PrimeArray(uint64_t bottom, uint64_t top)
  {
    this->bottom = bottom,
    this->top = top;
  }

  PrimeArray(){};
};


class ResetCounter{
private:
  uint16_t counter = 0;
public:
  inline void increment()
  {
    counter += 1;
    counter &= 511;
    if(counter == 0) cudaDeviceReset();
  }

  inline bool isReset(){return (bool) !counter;}
};

class x_Over_y{
private:
  uint64_t x;
public:
  __device__ __host__ x_Over_y(uint64_t x): x(x){}
  __device__ __host__ uint64_t operator() (uint64_t y) { return x / y;}
};
//
// __host__ __device__ static inline int64_t clzll(uint64_t x)
// {
//   uint64_t res;
// #ifdef __CUDA_ARCH__
//   res = __clzll(x);
// #else
//   asm("lzcnt %1, %0" : "=l" (res) : "l" (x));
// #endif
//   return res;
// }

__host__ __device__ inline uint64_t isqrt(uint64_t x)
{
  uint64_t m, y, b;
  int64_t t;

  m = 0x4000000000000000LL;
  y = 0;

  while(m != 0){
    b = y | m;
    y = y >> 1;
    t = (int64_t) (x | ~(x - b)) >> 31;
    x = x - (b & t);
    y = y | (m & t);
    m >>= 2;
  }
  return y;
}

#endif
