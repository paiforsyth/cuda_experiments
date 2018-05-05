#include "gpuerrchk.cuh"
#include "real.h"
#include "quad_tree_node.cuh" 
#include "points.cuh"
#include "parameters.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <random>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#define SECTION_SIZE 512
__global__ void ch13_quadtree_kernel(real* X, real* Y, int inputsize){
}

void ch13_quadtree(real* d_X, real* d_Y,int inputsize){
	ch13_quadtree_kernel<<<ceil(inputsize/ (real) SECTION_SIZE),SECTION_SIZE>>>(d_X,d_Y,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}




//currenty, my interpretation is the following:
//there are two points buffers, which alternate being active.  Within the currently active points buffer, the points associated with a particular node are in the range [node.points_begin(), node.points_end()).  These points are split amoung the various threads, whith one block of threads being assigned to each node.  thread i handles points with indexes starting from nodes.points_begin()+threadIdx.x and increase at intervals of Blockdim.x
__device__ bool check_num_points_and_depth(QuadTreeNode& node, Points* points, int num_points, Parameters params){
    if (params.depth == params.max_depth || num_points <= params.min_points_per_node){
        //stop the recurrsion, making sure points[0] contains all the points
        if (params.point_selector == 1){
            int it = node.points_begin();
            int end=node.points_end();
            for (it += threadIdx.x; it < end; it+=blockDim.x)
                if (it< end)
                    points[0].set_point(it, points[1].get_point(it));
        }
        return true;
    
    }
    return false;
}


//count the number of points in each quadrant
__device__ void count_points_in_children(const Points& in_points, int* smem, int range_begin, int range_end, float2 center){
     if(threadIdx.x < 4) smem[threadIdx.x] = 0;
     __syncthreads();
     //count the points in each quadraant, with each thread dealing with its own points
    for(int iter=range_begin+threadIdx.x; iter< range_end; iter+=blockDim.x){
        float2 p = in_points.get_point(iter);
        if(p.x < center.x && p.y >= center.y) //top left
           atomicAdd(&smem[0],1);
        if(p.x>= center.x && p.y >= center.y) //top right
           atomicAdd(&smem[1],1);
        if(p.x < center.x && p.y < center.y) //bottom left
            atomicAdd(&smem[2],1);
        if(p.x >= center.x && p.y <center.y) //bottom right
            atomicAdd(&smem[3],1);
    }
    __syncthreads();

}

__device__ void scan_for_offsets(int node_points_begin, int* smem){
    int* smem2 = &smem[4];
    if (threadIdx.x ==0 ){
        for(int i = 0; i<4; i++)
            smem2[i] = i==0 ? 0 : smem2[i-1] + smem[i-1];
        for(int i = 0; i<4; i++)
            smem2[i]+=node_points_begin;
    
    }
    __syncthreads();
}

//reorder points to group those in the same quadrant
//smem2[i] starts holding the number of points in quadrants ordered before
//quadrant i, and ends up recording the number of points in quadrants before or including quadrant i
__device__ void reorder_points(Points& out_points, const Points& in_points, int* smem, int range_begin, int range_end, float2 center){
    int* smem2 = &smem[4];
    for(int iter =range_begin +threadIdx.x; iter<range_end; iter+=blockDim.x){
        int dest;
        float2 p =in_points.get_point(iter);
        if(p.x < center.x && p.y >= center.y) //top left
           dest= atomicAdd(&smem2[0],1);
        if(p.x>= center.x && p.y >= center.y) //top right
           dest= atomicAdd(&smem2[1],1);
        if(p.x < center.x && p.y < center.y) //bottom left
           dest= atomicAdd(&smem2[2],1);
        if(p.x >= center.x && p.y <center.y) //bottom right
           dest= atomicAdd(&smem2[3],1);
        out_points.set_point(dest,p);
    }
    __syncthreads();
}

__device__ void prepare_children(QuadTreeNode* children, QuadTreeNode& node, const BoundingBox& bbox, int* smem, bool* active_nodes){
    //int child_offset = 4*node.id();  
    int child_offset = node.id();//I added this, since the above seemed to be wrong
    children[child_offset+0].set_id(4*node.id()+0);
    children[child_offset+1].set_id(4*node.id()+4);
    children[child_offset+2].set_id(4*node.id()+8);
    children[child_offset+3].set_id(4*node.id()+12);

    active_nodes[child_offset+0]=true;
    active_nodes[child_offset+1]=true;
    active_nodes[child_offset+2]=true;
    active_nodes[child_offset+3]=true;


    //points in bounding box:
    const float2& pmin = bbox.get_min();
    const float2& pmax = bbox.get_max();
    float2 center;
    bbox.compute_center(center); //does this work, given that bbox is const?

    children[child_offset+0].set_bounding_box(pmin.x, center.y, center.x, pmax.y ); //top left
    children[child_offset+1].set_bounding_box(center.x, center.y, pmax.x, pmax.y ); //top right
    children[child_offset+2].set_bounding_box(pmin.x, pmin.y, center.x, center.y ); //bottom left
    children[child_offset+3].set_bounding_box(center.x, pmin.y, pmax.x, center.y ); //bottom right

    //set the point ranges for the children
    children[child_offset + 0].set_range(node.points_begin(), smem[4 + 0]);
    children[child_offset + 1].set_range(smem[4 + 0], smem[4 + 1 ]);
    children[child_offset + 2].set_range(smem[4 + 1], smem[4 + 2 ]);
    children[child_offset + 3].set_range(smem[4 + 2], smem[4 + 3 ]);
    
}


__global__ void build_quad_tree_kernel(QuadTreeNode* nodes, Points* points, Parameters params,bool* active_nodes  ){
    __shared__ int smem[8];
    //idea: have only thread 0 display the point
    
    //the current node
    QuadTreeNode& node = nodes[blockIdx.x]; 
    node.set_id(node.id() + blockIdx.x);
    int num_points = node.num_points();
    //printf("hello!");
    cudaDeviceSynchronize();
    if(threadIdx.x==0){
       // printf("yo");
        printf("Depth: %d.  (depth-specific) node num: %d.  Point range: %d to %d. bbox : min= (%f,%f) max=(%f,%f).    \n",  params.depth, blockIdx.x, node.points_begin(), node.points_end(), node.bounding_box().get_min().x, node.bounding_box().get_min().y, node.bounding_box().get_max().x, node.bounding_box().get_max().y    );
    }
    cudaDeviceSynchronize();

    //check exit condition, moving points to first buffer as needed
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if(exit) return;
    
    const BoundingBox& bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center); //does this work, given that bbox is const?

    int range_begin = node.points_begin();
    int range_end = node.points_end();
    const Points& in_points = points[params.point_selector];
    Points& out_points = points[(params.point_selector +1) %2];

    //count points in each child
    count_points_in_children(in_points, smem, range_begin, range_end, center);
    if (threadIdx.x == 0){
        printf("Depth: %d.  Num: %d.  Points in children: (%d,%d,%d,%d) \n", params.depth, blockIdx.x, smem[0], smem[1], smem[2], smem[3] );
    }
    //compute reordering offset for each quadrant
    scan_for_offsets(node.points_begin(), smem);
    

    //reorder points (in other point buffer)
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    
    if (threadIdx.x == blockDim.x-1){
        printf("Depth: %d.  Num: %d launching child kernels\n", params.depth, blockIdx.x);
        QuadTreeNode* children = &nodes[params.num_nodes_at_this_level];
        bool* child_active_nodes = &active_nodes[params.num_nodes_at_this_level];
        prepare_children(children, node, bbox, smem,child_active_nodes);
        //launch child kernels
        build_quad_tree_kernel<<<4,blockDim.x, 8 * sizeof(int)>>>(children, points, Parameters(params, true), child_active_nodes);
    }



}

class RNG_functor{
    int ip;
    public:
        RNG_functor(int init_p){  
            ip = init_p; 
        }

    __host__ __device__
    float operator() (int dex) const 
        {
            thrust::default_random_engine generator;
            thrust::uniform_real_distribution<float> distribution(-1.0f,1.0f);
            generator.discard(ip+dex);
            return distribution(generator);
        }
};

int main(int argc, char **argv){
    std::cout << "Recieved " << argc << " arguments."<<std::endl;
    assert(argc == 4);
    //load paramters from command line
    const int num_points = atoi(argv[1]);
    const int max_depth = atoi(argv[2]);   
    const int min_points_per_node = atoi(argv[3]);
    std::cout << "Number of points is: "  << num_points << std::endl;
    std::cout << "Max depth is: "  << max_depth << std::endl;
    std::cout << "Min points per node is "  << min_points_per_node << std::endl;
    

    //allocate memory for points 
    cudaDeviceSynchronize();
    std::cout << "allocating device memory for points " << std::endl;
    thrust::device_vector<float> x_d0(num_points);
    thrust::device_vector<float> x_d1(num_points);
    thrust::device_vector<float> y_d0(num_points);
    thrust::device_vector<float> y_d1(num_points);

    //generate random points
    cudaDeviceSynchronize();
    std::cout << "generating random points " << std::endl;
    thrust::counting_iterator<int> count1_begin(0);
    thrust::counting_iterator<int> count2_begin(0);
    thrust::transform(count1_begin, count1_begin+num_points,x_d0.begin(),RNG_functor(0) );
    thrust::transform(count2_begin, count2_begin+num_points,y_d0.begin(),RNG_functor(42) );

    /*thrust::generate(
            thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(),y_d0.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(x_d0.end(), y_d0.end())),
            rng
            );
            */
    

     // host Points object whose uderlying data is the key  device_vectors
    cudaDeviceSynchronize();
    std::cout << "setting up points_init " << std::endl;
    cudaDeviceSynchronize();
    Points points_init[2];
    points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]),
                      thrust::raw_pointer_cast(&y_d0[0]) 
            );
    points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]),
             thrust::raw_pointer_cast(&y_d1[0]) 
            );

    //allocate Points objects on the device, refering to the same underlying data as above
    cudaDeviceSynchronize();
    std::cout << "allocating points object on device " << std::endl;
    cudaDeviceSynchronize();
    Points* points; 
    cudaMalloc( (void**) &points, 2*sizeof(Points) );
    cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice);
    
    //count the maximum number of nodes that could be needed
    int max_nodes = 0;
    for (int i=0, num_nodes_at_level=1; i<max_depth;++i, num_nodes_at_level*=4 )
        max_nodes += num_nodes_at_level;

    //alocate memory to store the tree
    cudaDeviceSynchronize();
    std::cout << "allocating tree " << std::endl;
    cudaDeviceSynchronize();
    QuadTreeNode root;
    root.set_range(0, num_points);
    QuadTreeNode* nodes;
    cudaMalloc((void**) &nodes, max_nodes*sizeof(QuadTreeNode) );
    cudaMemcpy(nodes, &root, sizeof(QuadTreeNode), cudaMemcpyHostToDevice );


    cudaDeviceSynchronize();
    std::cout << "allocating host_active_nodes " << std::endl;
    cudaDeviceSynchronize();
    bool* host_active_nodes = new bool[max_nodes];
    host_active_nodes[0]=true;
    for (int i=1; i< max_nodes; i++)
        host_active_nodes[i]=false;

    cudaDeviceSynchronize();
    std::cout << "allocating device active nodes " << std::endl;
    cudaDeviceSynchronize();
    bool* device_active_nodes;
    cudaMalloc((void**) &device_active_nodes, max_nodes*sizeof(bool)  );
    cudaDeviceSynchronize();
    std::cout << "copying host active nodes to device active nodes " << std::endl;
    cudaDeviceSynchronize();
    cudaMemcpy(device_active_nodes, &host_active_nodes[0], sizeof(bool)*max_nodes, cudaMemcpyHostToDevice);
    
    //set reucsion limit for cuda dynamic parallelism to max_depth
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,max_depth);

    //build the tree
    Parameters params(max_depth, min_points_per_node);
    const int NUM_THREADS_PER_BLOCK=32;
    const size_t smem_size=8*sizeof(int);
    std::cout << "Launching Kernel"<<std::endl;
    build_quad_tree_kernel<<<1, NUM_THREADS_PER_BLOCK,smem_size>>>(nodes, points, params, device_active_nodes);
	gpuErrchk(cudaPeekAtLastError());


    //inspect results
    //copy the nodes
    QuadTreeNode* host_nodes;
    host_nodes = (QuadTreeNode*) malloc( max_nodes * sizeof(QuadTreeNode) );
    cudaMemcpy(host_nodes,&nodes, max_nodes*sizeof(QuadTreeNode) , cudaMemcpyDeviceToHost);

    //copy the bools
    cudaMemcpy(host_active_nodes, &device_active_nodes, sizeof(bool)*max_nodes, cudaMemcpyDeviceToHost);

    

    for(int i=0; i<max_nodes; ++i)
        if(host_active_nodes[i]){
            std::cout<< "Printing Node number: "<< i << std::endl;
            //host_nodes[i].list_points(points[0]);
        }
    //free memory
    cudaFree(nodes);
    cudaFree(points);
}
