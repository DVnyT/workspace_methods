#include "../include/Index.hpp"
#include "../include/CudaUtils.hpp"
#include "../include/Tensor.hpp"
#include <array>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <chrono>

int main()
{
	cutensorCreate(&globalHandle);
	Tensor a(std::vector<int>({'i','j', 'k'}), std::vector<int64_t>({102, 4123, 23}));
	Tensor b(std::vector<int>({'j','k', 'l'}), std::vector<int64_t>({4123, 23, 22}));
	for (int i = 0; i < 9672558 ; i++)
	{
		a.getHostPtr()[i] = i;
	}
	for (int j = 0; j < 2086238; j++)
	{
		b.getHostPtr()[j] = j;
	}
	std::cout << "hello";
	auto start = std::chrono::high_resolution_clock::now();
	contractAB(a, b);
	auto end = std::chrono::high_resolution_clock::now();
	std:: cout << '\n' <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return 0;	
}

