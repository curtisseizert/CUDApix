#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#ifdef __CUDA_ARCH__
#include <math_functions.h>
#endif

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

#endif
