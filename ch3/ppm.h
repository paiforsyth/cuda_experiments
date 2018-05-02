#include <memory>
#include <fstream>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iterator>
#include <algorithm>
struct ppm{
	int width;
	int height;
    int* data;
	size_t size;
	
	~ppm() { delete[] data;}
	ppm(){}
	ppm(ppm& im){
		width=im.width;
		height=im.height;
		data=new int[height*width*3];
		std::copy(im.data,im.data+im.width*im.height*3,data);
	} 
	ppm(std::string filename){
		
		std::ifstream ifs;
		ifs.open(filename,std::ifstream::in);
		std::string dummy;
		ifs >> dummy;
		if (dummy != "P3") throw std::out_of_range("Wrong image type!!");
		ifs >> width;
		ifs >> height;
		ifs >> dummy;
		if (dummy!= "255") throw std:: out_of_range("Wrong image color depth!!");
		data=new int[height* width*3];
		for (int row=0; row< height; row++){
		for (int col=0; col<width*3;col+=3){
			ifs >> data[3*width*row+ col];
			ifs >> data[3*width*row+ col+1];
			ifs >> data[3*width*row+ col+2];
		}
		}
		ifs.close();
}
		void write(std::string filename){
			std::ofstream ofs(filename,std::ofstream::out);
			ofs << "P3\n";
			ofs << width << " " << height << "\n";
			ofs << "255\n";
			for (int row=0; row< height; row++){
			for (int col=0; col<width*3;col+=3){
				ofs << data[3*width*row+col]<< "\n";
				ofs << data[3*width*row+col+1]<< "\n";
				ofs << data[3*width*row+col+2]<< "\n";
			}
		//	ofs << "\n";
			}
			ofs.close();
		}
	void showdata(){
			for (int row=0; row< height; row++){
			for (int col=0; col<width*3;col+=3){
				std::cout << data[3*width*row+col]<< "\n";
				std::cout << data[3*width*row+col+1]<< "\n";
				std::cout << data[3*width*row+col+2]<< "\n";
			}
			//std::cout << "\n";
			}
		
	}	
};

