#include <cuda_runtime.h>
#include <stdio.h>
#include "norm.h"

int divup(int a, int b) {
  if (a % b) {
    return a / b + 1;
  } else {
    return a / b;
  }
}


__forceinline__ __device__ static float clampF(float x, float lower, float upper) {
  return x < lower ? lower : (x > upper ? upper : x);
}

__global__ void rgb_crop_norm_kernel(const uint8_t *src, float *dst, float fx_scale, float fy_scale, float fx_offset,
                                     float fy_offset, int src_width, int src_height, int dst_width, int dst_height,
                                     float mean_r, float mean_g, float mean_b, float scale_r, float scale_g,
                                     float scale_b) {
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x >= dst_width || dst_y >= dst_height) {
    return;
  }

  const int src_size = src_width * src_height;
  float src_x = dst_x * fx_scale + fx_offset;
  float src_y = dst_y * fy_scale + fy_offset;
  const int x1 = max(__float2int_rd(src_x), 0);
  const int y1 = max(__float2int_rd(src_y), 0);
  const int x1_read = x1;
  const int y1_read = y1;
  const int x2 = x1 + 1;
  const int y2 = y1 + 1;
  const int x2_read = min(x2, src_width - 1);
  const int y2_read = min(y2, src_height - 1);

  int idx11 = (y1_read * src_width + x1_read);
  int idx12 = (y1_read * src_width + x2_read);
  int idx21 = (y2_read * src_width + x1_read);
  int idx22 = (y2_read * src_width + x2_read);
  float weight11 = (x2 - src_x) * (y2 - src_y);
  float weight12 = (src_x - x1) * (y2 - src_y);
  float weight21 = (x2 - src_x) * (src_y - y1);
  float weight22 = (src_x - x1) * (src_y - y1);

  uchar3 src11 = make_uchar3(src[idx11], src[idx11 + src_size], src[idx11 + src_size * 2]);
  uchar3 src12 = make_uchar3(src[idx12], src[idx12 + src_size], src[idx12 + src_size * 2]);
  uchar3 src21 = make_uchar3(src[idx21], src[idx21 + src_size], src[idx21 + src_size * 2]);
  uchar3 src22 = make_uchar3(src[idx22], src[idx22 + src_size], src[idx22 + src_size * 2]);
  float3 out;
  out.x = src11.x * weight11 + src12.x * weight12 + src21.x * weight21 + src22.x * weight22;
  out.y = src11.y * weight11 + src12.y * weight12 + src21.y * weight21 + src22.y * weight22;
  out.z = src11.z * weight11 + src12.z * weight12 + src21.z * weight21 + src22.z * weight22;

  float out_r = (clampF(out.x, 0.0f, 255.0f) - mean_r) / scale_r;
  float out_g = (clampF(out.y, 0.0f, 255.0f) - mean_g) / scale_g;
  float out_b = (clampF(out.z, 0.0f, 255.0f) - mean_b) / scale_b;

  const int dst_idx = dst_y * dst_width + dst_x;
  dst[dst_idx] = out_r;
  dst[dst_idx + dst_width * dst_height] = out_g;
  dst[dst_idx + dst_width * dst_height * 2] = out_b;
  printf("%f %f %f", out_r, out_g, out_b);
}

void RGB_CropNorm(const uint8_t *src, float *dst, const RectI &roi, const Shape2DI &src_shape, const Shape2DI &dst_shape,
                  const Point3DF &mean, const Point3DF &scale, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid(divup(dst_shape.width, block.x), divup(dst_shape.height, block.y));
  float fx_scale = 1.0f * roi.GetWidth() / dst_shape.width;
  float fy_scale = 1.0f * roi.GetHeight() / dst_shape.height;
  float fx_offset = roi.l + 0.5f * fx_scale - 0.5f;
  float fy_offset = roi.t + 0.5f * fy_scale - 0.5f;
  rgb_crop_norm_kernel<<<grid, block, 0, stream>>>(src, dst, fx_scale, fy_scale, fx_offset, fy_offset, src_shape.width,
                                                   src_shape.height, dst_shape.width, dst_shape.height, mean.x, mean.y,
                                                   mean.z, scale.x, scale.y, scale.z);
}