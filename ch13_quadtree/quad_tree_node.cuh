#pragma once
#include "bounding_box.cuh"
#include "points.cuh"
class QuadTreeNode {
    public:
    int m_id; //identifier of the node
    int m_begin, m_end; //range of points for the node
    BoundingBox m_bounding_box;
    __host__ __device__ QuadTreeNode();

    __host__ __device__ int id() const;

    __host__ __device__ void set_id(int new_id);

    __host__ __device__  const BoundingBox& bounding_box() const;

    __host__ __device__  void set_bounding_box(float minx, float miny, float maxx, float maxy);


    __host__ __device__   int num_points() const;

    __host__ __device__  int points_begin() const;

    __host__ __device__  int points_end() const;

    __host__ __device__  void set_range(int begin, int end);

    __host__   void list_points(Points points);

};
