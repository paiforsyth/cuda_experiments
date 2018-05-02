#include "stringstore.h"
#include <string>
int main(){
	std::string hello="Hello World!";
	StringStore ss{hello};
	ss.print();
}
