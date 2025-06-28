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
	Tensor a(std::vector<int>({'i', 'j'}), std::vector<int64_t>({3, 3}));
	Tensor b(std::vector<int>({'j', 'k'}), std::vector<int64_t>({3, 3}));
	for (int i = 0; i < 9; i++)
	{
		a.getHostPtr()[i] = i + 1.0;
	}
	for (int j = 0; j < 9; j++)
	{
		b.getHostPtr()[j] = j + 5.0;
	}
	auto start = std::chrono::high_resolution_clock::now();
	Tensor c = contractAB(a, b);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::cout << c;
	std::cout << '\n' <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	
	
	auto start2 = std::chrono::high_resolution_clock::now();
	Tensor e = contractAB(a, b);
	auto end2 = std::chrono::high_resolution_clock::now();

	std::cout << '\n' << e;
	std::cout << '\n' <<  std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

	return 0;	
}

