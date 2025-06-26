#include "../include/Index.hpp"
#include <array>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
int main()
{
	Index a(1 , {"hello", "hi", "7letter", "8letters"}, 3);

	Index b(1 , {"hello", "hi", "7letter", "8letters"}, 3);
	Index c(1 , {"hello", "hi", "7letter", "8letters"}, 3);
	Index d(1 , {"hello", "hi", "7letter", "8letters"}, 3);
	Index e(1 , {"hello", "hi", "7letter", "8letters"}, 3);
	std::cout << a << b;
	std::cout << (a > b);
	return 0;	
}

