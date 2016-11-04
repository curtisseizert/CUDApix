#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "sieve/S2_hard_host.cuh"
#include "Gourdon/gourdonvariant.hpp"


int main()
{
    uint64_t x = pow(2, 60), y = 30737235, pi = 0;
    uint32_t c = 6;
    uint64_t s3 = 0;

    // s3 = S2hardHost::S2hard(x, y, c);
    //
    // std::cout << (int64_t)s3 << std::endl;
    //

    pi = GourdonVariant64::piGourdon(x, y, c);

    std::cout << "pi = " << pi << std::endl;

    return 0;
}
