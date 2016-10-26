#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#include "P2.cu"
#include "S3.cuh"
#include "phi.cuh"
#include "S0.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "sieve/lpf_mu.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"
#include "src/device128/device128.cu"

int main()
{
  uint128_t x = 1, p2 = 0;
  uint64_t y = 1015748248;

  x <<= 70;

  p2 = P2(x, y);

  return 0;
}
