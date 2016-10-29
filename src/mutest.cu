#include <stdint.h>

#include "sieve/lpf_mu.cuh"

// Merten's function to test the output of the mobius function
int64_t mutest(uint64_t top)
{
  int8_t * h_mu = gen_h_mu((uint64_t)0, (uint64_t)top);
  int64_t sum = 0;

  for(uint64_t i = 0; i < top; i++){
    if(i % 2 == 0){
      if(i % 4 != 0)
        sum -= h_mu[i/4];
    }else{
      sum += h_mu[i/2];
    }
  }
  return sum;
}
