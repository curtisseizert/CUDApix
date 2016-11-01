#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "general/tools.hpp"

template<typename T, typename U>
void dispDeviceArray(T * d_a, U numElements)
{
  int64_t total = 0;
  T * h_a = (T *)malloc(numElements * sizeof(T));
  cudaMemcpy(h_a, d_a, numElements * sizeof(T), cudaMemcpyDeviceToHost);

  for(U i = 0; i < numElements; i++){
    std::cout << i << "\t" << h_a[i] << std::endl;
    total += h_a[i];
  }

  std::cout << "Total\t" << total << std::endl;

  free(h_a);
}
template void dispDeviceArray<int16_t, uint32_t>(int16_t * d_a, uint32_t numElements);
template void dispDeviceArray<int32_t, uint32_t>(int32_t * d_a, uint32_t numElements);
template void dispDeviceArray<int64_t, uint32_t>(int64_t * d_a, uint32_t numElements);
template void dispDeviceArray<uint64_t, uint32_t>(uint64_t * d_a, uint32_t numElements);

template<typename T, typename U>
void dispDevicePartialSums(T * d_a, U numElements, U dimx)
{
  U dimy = numElements/dimx;
  int64_t total = 0;
  T * h_a = (T *)malloc(numElements * sizeof(T));
  int64_t * h_sums = (int64_t *)malloc(dimy * sizeof(int64_t));

  for(U i = 0; i < dimy; i++) h_sums[i] = 0;

  cudaMemcpy(h_a, d_a, numElements * sizeof(T), cudaMemcpyDeviceToHost);

  for(U i = 0; i < numElements; i++){
    total += h_a[i];
    h_sums[i/dimx] += h_a[i];
  }

  for(U i = 0; i < dimy; i++)
    std::cout <</* i << "\t" <<*/ h_sums[i] << std::endl;

  std::cout << "total:\t" << total << std::endl;

  free(h_a);
}

template void dispDevicePartialSums<int16_t, uint32_t>(int16_t * d_a, uint32_t numElements, uint32_t dimx);
template void dispDevicePartialSums<int32_t, uint32_t>(int32_t * d_a, uint32_t numElements, uint32_t dimx);
template void dispDevicePartialSums<int64_t, uint32_t>(int64_t * d_a, uint32_t numElements, uint32_t dimx);
template void dispDevicePartialSums<uint64_t, uint32_t>(uint64_t * d_a, uint32_t numElements, uint32_t dimx);
