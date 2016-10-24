#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#include "S3.cuh"
#include "phi.cuh"
#include "S0.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "sieve/lpf_mu.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"

typedef unsigned __int128 uint128_t;

int main()
{
  uint64_t x, y, p2, pix, piy;
  int64_t s3, * d_phi;
  x = pow(10,12);
  y = 10000;
  uint32_t c = 6;

  Phi phi(x/y, y);

  d_phi = phi.generateRange()

  return 0;
}
