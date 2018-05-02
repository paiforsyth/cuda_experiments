#include "points.cuh"
#include <iostream>
__host__ __device__ Points::Points() : m_x(NULL), m_y(NULL){}
    
__host__ __device__ Points::Points(float* x, float* y): m_x(x), m_y(y){}

__host__ __device__ __forceinline__ float2 Points::get_point(int idx) const{
    return make_float2(m_x[idx], m_y[idx]);
}

 __host__ void Points::print_point(int idx) const{
   float2 p = get_point(idx);
   std::cout << "x: " <<  p.x << " y:" << p.y << std::endl;
 }
