#include "parameters.cuh"
__host__ __device__ Parameters::Parameters(int max_depth, int min_points_per_node):point_selector(0), num_nodes_at_this_level(1), depth(0), max_depth(max_depth), min_points_per_node(min_points_per_node){}

__host__ __device__ Parameters(const Parameters& params, bool):
    point_selector((params.point_selector+1)%2),
    num_nodes_at_this_level(4*params.num_nodes_at_this_level),
    depth(params.depth+1),
    max_depth(params.max_depth),
    min_points_per_node(params.min_points_per_node) {}
