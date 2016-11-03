#include <stdint.h>
#include <cuda_runtime.h>
#include "sieve/S2_hard_device.cuh"

#ifndef _S2_HARD_HOST
#define _S2_HARD_HOST

class S2hardHost{
private:
  S2data_64 * data;
  c_data64 h_cdata;
  uint16_t numStreams_ = 5, sieveWords_ = 3072;
  cudaStream_t stream[5];

public:
  S2hardHost(uint64_t x, uint64_t y, uint16_t c);
  ~S2hardHost();

  void makeData(uint64_t x, uint64_t y, uint16_t c);

  void setupNextIter();

  void allocate();
  void deallocate();
  void zero();
  void transferConstants();

  int64_t launchIter();


  static uint64_t S2hard(uint64_t x, uint64_t y, uint16_t c);

};



#endif
