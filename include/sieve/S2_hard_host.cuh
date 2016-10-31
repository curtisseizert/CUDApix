#include <stdint.h>
#include "sieve/S2_hard_device.cuh"

#ifndef _S2_HARD_HOST
#define _S2_HARD_HOST

class S2hardHost{
private:
  S2data_64 * data;
public:
  S2hardHost(uint64_t x, uint64_t y, uint16_t c);
  ~S2hardHost();

  void makeData(uint64_t x, uint64_t y, uint16_t c);

  void allocate();
  void deallocate();

  int64_t launchIter();

  static uint64_t S2hard(uint64_t x, uint64_t y, uint16_t c);

};



#endif
