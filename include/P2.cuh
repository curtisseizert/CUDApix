/*
P2.cuh - header file for computing second partial sieve function

Curtis Seizert <cseizert@gmail.com>

*/

#include <gmpxx.h>
#ifdef __CUDA_ARCH__
#include "uint128_t.cu"
#endif

#ifndef _P2
#define _P2

#define THREADS_PER_BLOCK 256

uint64_t P2(uint64_t x, uint64_t y);
mpz_class P2(uint128_t x, uint64_t sqrt_x, uint64_t y);
mpz_class P2(mpz_class x, mpz_class y);


struct PrimeArray{
  uint64_t * h_primes, * d_primes, bottom, top, pi_bottom, pi_top, diff_pi;
  size_t len;
};

class ResetCounter{
private:
  uint16_t counter = 0;
public:
  inline void increment();
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
