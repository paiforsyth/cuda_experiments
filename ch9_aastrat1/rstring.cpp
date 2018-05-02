#include <string>
#include <random>
#include <iostream>
const std::string alpha("abcdefghijklmnopqrstuvwxyz");

std::string rstring(size_t length){
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dis(0, 25);
	std::string out;
	for(int i=0;i<length;++i){
			out.append(1,alpha[dis(gen)]);
	}
	return out;	
}

/*int main(){
	std::cout << rstring(10)  <<"\n";
}
*/
