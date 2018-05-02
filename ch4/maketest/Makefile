CXXFLAGS = -std=c++14 

testmain : testmain.cpp ./build/stringstore.o
	$(CXX) $(CXXFLAGS) ./build/stringstore.o testmain.cpp -o testmain

./build/stringstore.o : stringstore.h 
	$(CXX) $(CXXFLAGS) -c -o ./build/stringstore.o  stringstore.cpp

