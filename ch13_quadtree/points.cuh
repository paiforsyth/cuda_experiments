#pragma once
class Points{
    
    public:
    float* m_x;
    float* m_y;

    __host__ __device__ Points();
    
    __host__ __device__ Points(float* x, float* y);

    __host__ __device__  float2 get_point(int idx) const;

    __host__ void print_point_d_2_h(int idx) const;

    __host__ __device__  void set_point(int idx, const float2& p);
    __host__ __device__  void set(float* x , float* y); 
        

};
