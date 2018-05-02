#include "stringstore.h"
#include <string>
#include <iostream>
StringStore::StringStore(std::string in):data(in){}
void StringStore::print(){ std::cout << data << '\n' <<std::flush; }
