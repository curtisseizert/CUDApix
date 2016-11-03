#include <stdint.h>

template<typename T, typename U>
void dispDeviceArray(T * d_a, U numElements);

template<typename T, typename U>
void dispDevicePartialSums(T * d_a, U numElements, U dimx);

template<typename T>
T lowerBound(T * a, T lo, T hi, T value);

template<typename T>
T upperBound(T * a, T lo, T hi, T value);
