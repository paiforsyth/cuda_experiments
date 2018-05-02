class BoundingBox{
    float2 m_p_min;
    float2 m_p_max;

    public:
    __host__ __device__ BoundingBox();
    __host__ __device__ void compute_center(float2 &center) const;
    __host__ __device__ float2& get_max() const;
    __host__ __device__ float2& get_min() const;
    __host__ __device__ bool contains(float2& p) const;
    __host__ __device__ void set(float min_x, float min_y, float max_x, float_max_y);

};
