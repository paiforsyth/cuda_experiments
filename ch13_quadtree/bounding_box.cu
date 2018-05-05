#include "bounding_box.cuh"
__host__ __device__ BoundingBox::BoundingBox(){
    m_p_min = make_float2(-1.0f, -1.0f);
    m_p_max = make_float2(1.0f, 1.0f);
}

__host__ __device__ void BoundingBox::compute_center(float2 &center) const{
    center.x = (m_p_min.x + m_p_max.x)/2;
    center.y = (m_p_min.y + m_p_max.y )/2;    
}
__host__ __device__ const  float2& BoundingBox::get_max() const{
        return m_p_max;    
}
__host__ __device__ const float2& BoundingBox::get_min() const{
    return m_p_min;    
}
__host__ __device__ bool BoundingBox::contains(float2& p) const{
    return p.x >= m_p_min.x && p.y >= m_p_min.y && p.x <= m_p_max.x && p.y <= m_p_max.y;
}

__host__ __device__ void BoundingBox::set(float min_x, float min_y, float max_x, float max_y){
    m_p_min.x = min_x;
    m_p_min.y = min_y;
    m_p_max.x = max_x;
    m_p_max.y = max_y;
}


