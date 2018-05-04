#include "points.cuh"
#include <iostream>
__host__ __device__ Points::Points() : m_x(NULL), m_y(NULL){}
    
__host__ __device__ Points::Points(float* x, float* y): m_x(x), m_y(y){}

__host__ __device__  float2 Points::get_point(int idx) const{
    return make_float2(m_x[idx], m_y[idx]);
}

 __host__ void Points::print_point_d_2_h(int idx) const{
     //hack
   float* host_mx;
   cudaMemcpy(host_mx, mx,sizeof(float)*(idx+1), cudaMemcpyDeviceToHost);
   float* host_my;
   cudaMemcpy(host_my, my,sizeof(float)*(idx+1), cudaMemcpyDeviceToHost);
   std::cout << "x: " <<  host_mx[idx] << " y:" << host_my_[idx] << std::endl;
 }
 __host__ __device__  void Points::set_point(int idx, const float2& p){
    m_x[idx] = p.x;
    m_y[idx] = p.y; 
}
//set the actual pointers
__host__ __device__  void Points::set(float* x , float* y){
    m_x= x;
    m_y= y;
}
