#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

class Index
{
private:
	int order{0};
	std::vector<std::string> tags;
	int prime{0};
	uintptr_t hash = reinterpret_cast<uintptr_t>(this);	// std::cout << hash; will print the address!

public:
	Index() = default;

	Index(int n)
	: order(n)
	{}

	Index(const Index&)
	{}

	Index(int n, std::vector<std::string>* pNames)
	: order(n), tags(*pNames)
	{}

	Index(int n, int p)
	: order(n), prime(p)
	{}

	Index(int n, std::vector<std::string>* pNames, int p)
	: order(n), tags(*pNames), prime(p)
	{}

	auto operator<=>(const Index& other) const = default;  // wtf
	
	void Prime()
	{
		this->prime++;  // TODO: move this to the .cpp file it belongs to (only declarations here)
	}
};



