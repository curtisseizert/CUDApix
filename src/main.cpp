#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#include "mutest.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "sieve/phisieve_host.cuh"
#include "S3.cuh"
#include "phi.cuh"
#include "S0.cuh"
#include "uint128_t.cuh"
#include "sieve/lpf_mu.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"


int main()
{
    uint64_t x = pow(10, 12), y = 10000;
    uint32_t c = 6;
    uint64_t s3 = 0;

    s3 = S3(x, y, c);

    std::cout << (int64_t)s3 << std::endl;

    return 0;
}
