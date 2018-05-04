#include "quad_tree_node.cuh"

__host__ __device__ QuadTreeNode::QuadTreeNode(): m_id(0), m_begin(0), m_end(0) {}
    
__host__ __device__ int QuadTreeNode::id() const{
    return m_id;
}

__host__ __device__ void QuadTreeNode::set_id(int new_id){
    m_id= new_id;
}


__host__ __device__  const BoundingBox& QuadTreeNode::bounding_box() const{
    return m_bounding_box;
}

__host__ __device__  void QuadTreeNode::set_bounding_box(float minx, float miny, float maxx, float maxy){
    m_bounding_box.set(minx, miny, maxx, maxy);
}


__host__ __device__   int QuadTreeNode::num_points() const{
    return m_end - m_begin;
}

__host__ __device__  int QuadTreeNode::points_begin() const{
    return m_begin;
}

__host__ __device__  int QuadTreeNode::points_end() const{
    return m_end;
}

__host__ __device__  void QuadTreeNode::set_range(int begin, int end){
    m_begin = begin;
    m_end = end;
}


