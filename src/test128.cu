#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <gmpxx.h>

#include "P2.cu"
#include "S3.cuh"
#include "phi.cuh"
#include "S0.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "sieve/lpf_mu.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"

int main()
{
  mpz_class total;
  uint128_t x;
  uint64_t sqrt_x, y;
  sqrt_x = 1152921504606846900;
  std::cout << sqrt_x << std::endl;
  y = 1234567897;

  x = uint128_t::mul128(sqrt_x, sqrt_x);

  std::cout << sqrt_x << " " << uint128_t::sqrt(x) << std::endl;

  //total = P2(x, y, sqrt_x);


  return 0;
}
