#include <stdint.h>
#include <stdio.h>

#ifndef _PITABLE
#define _PITABLE

uint32_t * get_d_piTable(uint32_t hi);

__global__ void transposePrimes(uint32_t * d_primes, uint32_t * d_pitable, uint32_t top, size_t len);
__global__ void transposePrimes(uint64_t * d_primes, uint32_t * d_pitable, uint64_t base, uint64_t range, size_t len, uint64_t pi_base);

class PiTable{
private:
  uint64_t base = 0, range = 0, allocatedRange = 0, pi_base = 0, bottom = 0;
  size_t len, free_mem, tot_mem;
  bool isTableCurrent = 0, isPiCurrent = 0;

  uint32_t * d_pitable = NULL;

  void calc_pi_base();
public:

  PiTable(){}
  PiTable(uint64_t range);
  PiTable(uint64_t base, uint64_t bottom, uint64_t range = 0);

  ~PiTable();

  uint64_t get_base(){return base;}
  uint64_t get_range(){return range;}
  uint64_t get_bottom(){return bottom;}
  uint64_t get_pi_base();

  void set_base(uint64_t base);
  void set_range(uint64_t range);
  void set_pi_base(uint64_t pi_base);
  void set_bottom(uint64_t bottom){this-> bottom = bottom;}

  uint64_t getMaxRange();

  void allocate();
  void reallocate();
  uint64_t setMaxRange();

  uint32_t * getNextUp();
  uint64_t getNextBaseUp(){return base + range;}
  uint32_t * getNextDown();
  uint64_t getNextBaseDown(){return std::max((int64_t)bottom, (int64_t)(base - range));}
  uint32_t * getCurrent();

};

#endif
