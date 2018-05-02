#include <ctime>
#include <ratio>
#include <chrono>
template<typename FuncType>
void time(FuncType f){
  using namespace std::chrono;
  std::cout<< "initiating timing \n" << std::flush;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  
  f();

  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.";
  std::cout << std::endl;  
}
