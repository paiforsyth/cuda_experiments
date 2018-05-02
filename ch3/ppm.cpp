#include "ppm.h"
int main(){
	ppm football("football.ppm");
	ppm football2(football);	
	football2.write("football2.ppm");
}

