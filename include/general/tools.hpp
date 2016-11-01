#include <stdint.h>

template<typename T, typename U>
void dispDeviceArray(T * d_a, U numElements);

template<typename T, typename U>
void dispDevicePartialSums(T * d_a, U numElements, U dimx);
