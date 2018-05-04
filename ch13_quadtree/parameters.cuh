class Parameters {
    public:
    int point_selectior;

    int num_nodes_at_this_level;
    
    int depth;

    const int max_depth;

    const int min_ponts_per_node;
    __host__ __device__  Parameters(int max_depth, int min_points_per_node);

    __host__ __device__ Parameters(const Parameters& params, bool);//constructor that changes values for next iteration.  Role of bool seems to be merely to distinguish this from the default copy constructor

};
