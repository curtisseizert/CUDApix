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
  // mpz_class x, y, total;
  // x = pow(2,64);
  // y = 97125767;//231112254739;//1015748248;

  uint64_t x, y, p2, pix, piy;// s1_trivial, totalV, s0;
  int64_t s3;
  x = pow(10,12);
  y = 10000;
  uint32_t c = 6;
  std::cout << "y = " << y << std::endl;
  piy = CudaSieve::countPrimes(0, y, 0);

  p2 = P2(x,y);
  std::cout << "P2  = " << p2 << std::endl;
  //
  // s1_trivial = S1_trivial(x,y);
  // std::cout << "S1_trivial = " << s1_trivial << std::endl;
  //
  // totalV = V(x,y);
  // std::cout << "V = " << totalV << std::endl;
  //
  // uint64_t pi = s1_trivial + totalV + y - 1 - totalP2;
  //
  // std::cout << "\n" << pi << std::endl;

  // s0 = S0(x, y);
  // std::cout << s0 << std::endl;

  s3 = S3(x, y, c);

  std::cout << s3 << std::endl;

  pix = piy + s3 - 1 - p2;

  std::cout << "pi(x) = " << pix << std::endl;

  return 0;
}
