#include <stdint.h>

#pragma once

__global__ void isGreaterThan(uint32_t num, uint32_t * d_arrayIn, uint32_t * d_arrayOut, uint32_t value);
__global__ void addOneToStart(uint32_t * a);


class greater_than_n{
private:
  uint32_t n;
public:
  __device__ __host__ greater_than_n(uint32_t n): n(n){}
  __device__ __host__ bool operator() (uint32_t y) { return (y > n) ? 1 : 0;}
};

class Phi{
private:
  uint32_t * d_lpf = NULL, * h_cache, max_, p_max_, c_;
  uint32_t * d_primeList, * h_primeList, primeListLength;
public:

  uint32_t * get_d_lpf(){return d_lpf;}
  uint32_t * get_d_primeList(){return d_primeList;}
  uint32_t * get_h_primeList(){return h_primeList;}
  uint32_t get_primeListLength(){return primeListLength;}

  Phi(uint32_t max);
  Phi(uint32_t max, uint32_t p_max);
  ~Phi();

  uint64_t phi(uint32_t x, uint32_t a);
  void makeModCache(uint32_t pi_max);

  uint32_t * generateRange(uint32_t num, uint32_t a);
};
