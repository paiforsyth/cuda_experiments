#include <iostream>
#include "bounding_box.cuh"
int main(){
    BoundingBox box;        
    float2 p1 = make_float2(0.5f,0.5f);
    float2 p2 = make_float2(10.0f, 10.0f);

    std::cout << "Does point 1 lie in the box? " << box.contains(p1) << std::endl;
    std::cout << "Does point 2 lie in the box? " <<< box.contains(p2) << std::endl;

}
