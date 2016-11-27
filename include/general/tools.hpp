#include <stdint.h>
#include <iostream>
#include <cuda_runtime.h>

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
    std::cout << i << "\t" << h_sums[i] << std::endl;

  std::cout << "total:\t" << total << std::endl;

  free(h_a);
}



template<typename T, typename U, typename V, typename W>
T lowerBound(T * a, U lo, V hi, W value)
{
  do{
    T mid = (lo + hi) / 2;
    if(a[mid] < value) lo = mid;
    if(a[mid] >= value) hi = mid;
  }while(lo + 1 < hi);
  return lo;
}

template<typename T, typename U, typename V, typename W>
T upperBound(T * a, U lo, V hi, W value)
{
  do{
    T mid = (lo + hi) / 2;
    if(a[mid] <= value) lo = mid;
    if(a[mid] > value) hi = mid;
  }while(lo + 1 < hi);
  return hi;
}
