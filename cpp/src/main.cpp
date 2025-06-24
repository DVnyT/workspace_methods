#include "../include/Index.hpp"
#include <array>
#include <vector>
#include <iostream>
int main()
{
	std::vector<std::string_view> tags = { "tag1", "tag2" };
	Index a(3, tags, 2);
	std::cout << a;
	return 0;
}

