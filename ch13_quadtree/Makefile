FLAGS = -std=c++14 -I~/Dropbox/src -arch=sm_35 -rdc=true
LIBS = -lcudadevrt






ch13_quadtree : ch13_quadtree.cu ./build/points.o ./build/bounding_box.o  ./build/quad_tree_node.o   ./build/parameters.o
	nvcc $(FLAGS)  -o ch13_quadtree ch13_quadtree.cu ./build/points.o ./build/bounding_box.o  ./build/quad_tree_node.o  ./build/parameters.o $(LIBS)

points_test : points_test.cu ./build/points.o points.cuh
	nvcc $(FLAGS) -o points_test points_test.cu ./build/points.o $(LIBS)


./build/points.o : points.cu
	nvcc $(FLAGS) -c -o ./build/points.o points.cu $(LIBS)


bounding_box_test : bounding_box_test.cu ./build/bounding_box.o bounding_box.cuh
	nvcc $(FLAGS) -o bounding_box_test bounding_box_test.cu ./build/bounding_box.o $(LIBS)


./build/bounding_box.o : bounding_box.cu
	nvcc $(FLAGS) -c -o ./build/bounding_box.o bounding_box.cu  $(LIBS)
 

./build/quad_tree_node.o : quad_tree_node.cu quad_tree_node.cuh bounding_box.cuh
	nvcc $(FLAGS) -c -o ./build/quad_tree_node.o quad_tree_node.cu  $(LIBS)


./build/parameters.o : parameters.cu parameters.cuh
	nvcc $(FLAGS) -c -o ./build/parameters.o parameters.cu  $(LIBS)

