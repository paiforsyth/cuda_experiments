#include <iostream>
void vecAdd(float* a, float* b, float*c, int n){
	for (int i = 0; i< n; i++) c[i]=a[i]+b[i]; 
}
    
int main(){
	float a[]={1,2,3,4,5,6};
	float b[]={1,2,3,4,5,6};
	float c[6];
	int n {6};
	vecAdd(a,b,c,n);
	for (int i=0;i<6;i++){
		std::cout << c[i] << '\n';
	}
}
