
#pragma once
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef __uint8_t uint8_t;
typedef __int8_t int8_t;

#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include "geo.h"



int divup(int a, int b);

void RGB_CropNorm(const uint8_t* src, float* dst, const RectI& roi, const Shape2DI& src_shape, const Shape2DI& dst_shape,
                  const Point3DF& mean, const Point3DF& scale, cudaStream_t stream);