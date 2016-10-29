#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "sieve/phisieve_device.cuh"
#include "sieve/phisieve_host.cuh"

void Phisieve::init(uint16_t c)
{
  sieveCountInit<<<blocks, threads>>>(d_sieve, d_count, bstart, c);
  cudaDeviceSynchronize();

  thrust::inclusive_scan(thrust::device, d_count, d_count + blockSize, d_count);
  cudaDeviceSynchronize();
  
  countFinit<<<blocks, threads>>>(d_count);
}
