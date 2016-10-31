#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "general/tools.hpp"

template<typename T, typename U>
void dispDeviceArray(T * d_a, U numElements)
{
  T * h_a = (T *)malloc(numElements * sizeof(T));
  cudaMemcpy(h_a, d_a, numElements * sizeof(T), cudaMemcpyDeviceToHost);

  for(U i = 0; i < numElements; i++)
    std::cout << i << "\t" << h_a[i] << std::endl;

  free(h_a);
}
template void dispDeviceArray<int16_t, uint32_t>(int16_t * d_a, uint32_t numElements);
template void dispDeviceArray<int32_t, uint32_t>(int32_t * d_a, uint32_t numElements);
template void dispDeviceArray<int64_t, uint32_t>(int64_t * d_a, uint32_t numElements);
template void dispDeviceArray<uint64_t, uint32_t>(uint64_t * d_a, uint32_t numElements);
