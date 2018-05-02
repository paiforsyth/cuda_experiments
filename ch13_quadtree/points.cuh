#pragma once
class Points{
    float* m_x;
    float* m_y;

    public:
    __host__ __device__ Points();
    
    __host__ __device__ Points(float* x, float* y);

    __host__ __device__ __forceinline__ float2 get_point(int idx) const;

};
