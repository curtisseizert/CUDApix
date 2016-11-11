#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <uint128_t.cuh>
#include <string>
#include <CUDASieve/cudasieve.hpp>
#include <cuda_profiler_api.h>

#include "ordinary.cuh"
#include "general/leafcount.hpp"
#include "general/tools.hpp"
#include "Gourdon/gourdonvariant.hpp"
#include "P2.cuh"
#include "pitable.cuh"

uint128_t calc(char * argv);
uint64_t echo(char * argv);
double getAlpha(uint64_t x);
double getAlpha(uint128_t x);


int main(int argc, char ** argv)
{
    double alpha;
    uint64_t x = pow(10, 12);
    // uint128_t x = (uint128_t)1 << 70;
    uint64_t y = 0, pi = 0;
    uint32_t c = 6;
    uint64_t s3 = 0;

    if(argc > 4){
      std::cerr << "Unexpected input." << std::endl;
      return 1;
    }else{
      for(uint16_t i = 1; i < argc; i++){
        if(i == 1) x = echo(argv[1]);
        if(i == 2) y = echo(argv[2]);
        if(i == 3) c = echo(argv[3]);
      }
    }

    if( y == 0){
      alpha = getAlpha(x);
      y = alpha * cbrt(x);
    }

    // if(argc > 4){
    //   std::cerr << "Unexpected input." << std::endl;
    //   return 1;
    // }else{
    //   for(uint16_t i = 1; i < argc; i++){
    //     if(i == 1) x = calc(argv[1]);
    //     if(i == 2) y = echo(argv[2]);
    //     if(i == 3) c = echo(argv[3]);
    //   }
    // }
    //
    // if(y == 0){
    //   alpha = getAlpha(x);
    //   y = uint128_t::sqrt(x);
    //   y = pow(y, (double)2.0/3.0);
    //   y *= alpha;
    // }

    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    std::cout << "z = " << x/y << std::endl;
    std::cout << "c = " << c << std::endl;

    uint64_t p0 = ordinary(x, y, c);
    std::cout << p0 << std::endl;

    // pi = GourdonVariant64::piGourdon(x, y, c);
    // std::cout << "pi = " << pi << std::endl;
    //
    // countEasyGourdon(x);

    cudaDeviceReset();
    return 0;
}

uint128_t calc(char * argv) // for getting values bigger than the 32 bits that system() will return;
{
  uint128_t value;
  size_t len = 0;
  char * line = NULL;
  FILE * in;
  char cmd[256];

  sprintf(cmd, "calc %s | awk {'print $1'}", argv);

  in = popen(cmd, "r");
  getline(&line, &len, in);
  std::string s = line;

  value = uint128_t::stou128_t(s);

  return value;
}

uint64_t echo(char * argv) // for getting values bigger than the 32 bits that system() will return;
{
  uint64_t value;
  size_t len = 0;
  char * line = NULL;
  FILE * in;
  char cmd[256];

  sprintf(cmd, "echo $((%s))", argv);

  in = popen(cmd, "r");
  getline(&line, &len, in);
  value = atol(line);

  return value;
}


// I "borrowed" this from Kim Walisch's primecount to facilitate comparing results
double getAlpha(uint64_t x)
{
  double alpha;
  double x2 = (double) x;

// use default alpha if no command-line alpha provided
  if (x2 <= 1e21)
  {
    double a = 0.000711339;
    double b = -0.0160586;
    double c = 0.123034;
    double d = 0.802942;
    double logx = log(x2);

    alpha = a * pow(logx, 3) + b * pow(logx, 2) + c * logx + d;
  }
  else
  {
    // Because of CPU cache misses sieving (S2_hard(x) and P2(x))
    // becomes the main bottleneck above 10^21 . Hence we use a
    // different alpha formula when x > 10^21 which returns a larger
    // alpha which reduces sieving but increases S2_easy(x) work.
    double a = 0.00149066;
    double b = -0.0375705;
    double c = 0.282139;
    double d = 0.591972;
    double logx = log(x2);

    alpha = a * pow(logx, 3) + b * pow(logx, 2) + c * logx + d;
  }

  return alpha;
}

double getAlpha(uint128_t x)
{
  double alpha;
  double x2 = (double) uint128_t::sqrt(x);

// use default alpha if no command-line alpha provided
  if (x2 <= 31622776601)
  {
    double a = 0.000711339;
    double b = -0.0160586;
    double c = 0.123034;
    double d = 0.802942;
    double logx = 2 * log(x2);

    alpha = a * pow(logx, 3) + b * pow(logx, 2) + c * logx + d;
  }
  else
  {
    // Because of CPU cache misses sieving (S2_hard(x) and P2(x))
    // becomes the main bottleneck above 10^21 . Hence we use a
    // different alpha formula when x > 10^21 which returns a larger
    // alpha which reduces sieving but increases S2_easy(x) work.
    double a = 0.00149066;
    double b = -0.0375705;
    double c = 0.282139;
    double d = 0.591972;
    double logx = 2 * log(x2);

    alpha = a * pow(logx, 3) + b * pow(logx, 2) + c * logx + d;
  }

  return alpha;
}
