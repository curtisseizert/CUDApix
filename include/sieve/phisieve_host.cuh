#include <stdint.h>

// phi(n,a) = d_count_[(1 + n)/2]


class Phisieve{
private:
  uint32_t maxPrime_, *h_primeList_ = NULL, *d_primeList_ = NULL, primeListLength_;
  uint32_t blockSize_ = 1u << 28; // == #bits in the sieve to cover this block
  uint32_t blocks_, threads_; // these need to be updated when changing blockSize
  uint32_t * d_sieve_ = NULL, * d_count_ = NULL;
  uint64_t bstart_ = 0;
  uint32_t a_current_;
  uint16_t cutoff_ = 12; // this is the cutoff between types of sieve, i.e.
                         // cutoff = pi(largest small prime).  This needs to be
                         // the same as that in the list of sieving primes coming
                         // from CUDASieve as well as in these functions
  void init();

public:
  Phisieve(uint32_t maxPrime);
  Phisieve(uint32_t maxPrime, uint32_t blockSize);

  ~Phisieve();

  void firstSieve(uint16_t c);

  void markNext();

  void updateCount();

  uint32_t * getCountHost();
  uint32_t * getCountDevice(){return d_count_;}

  uint32_t get_a_current(){return a_current_;}

  void set_a_current(uint32_t a){a_current_ = a;}


};
