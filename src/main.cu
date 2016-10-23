#include <iostream>
#include <math.h>
#include <gmpxx.h>

#include "S0.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "sieve/lpf.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"

typedef unsigned __int128 uint128_t;

int main()
{
  // mpz_class x, y, total;
  // x = pow(2,64);
  // y = 97125767;//231112254739;//1015748248;

  uint64_t x, y, totalP2, s1_trivial, totalV, s0;
  x = pow(10,12);
  y = 69482;
  std::cout << "y = " << y << std::endl;
  //
  // totalP2 = P2(x,y);
  // std::cout << "P2  = " << totalP2 << std::endl;
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

  s0 = S0(x, y);
  std::cout << s0 << std::endl;

  // cudaMallocHost(&h_mu, top * sizeof(int8_t)/2);
  // cudaMemcpy(h_mu, d_mu, top * sizeof(int8_t)/2, cudaMemcpyDeviceToHost);
  //
  // for(uint32_t i = m_n/4; i < m_n/2; i++) mertens += h_mu[i];
  //
  // for(uint32_t i = 0; i < 50; i ++) std::cout << 2*i+1 << "\t" << (int) h_mu[i] << std::endl;
  //
  //
  // std::cout << "\n" << mertens << std::endl;

  return 0;
}
