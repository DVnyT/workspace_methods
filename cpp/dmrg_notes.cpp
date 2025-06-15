# include <chrono>
# include <complex>
# include <iostream>
# include <vector>
# include <map>
# include <unordered_map>
# include <tuple>

# include <cuda_runtime.h>
# include <cutensor.h>

/*
 * We want to create some classes that handle internal cuTENSOR boilerplate, and eventually write an implementation for* the DMRG algorithm.
 * Our goals =>
 * 1. Create an Index class that models functionality after ITensor; 
 *    a) A variable for the 'order' of the Index (cuTENSOR calls this the extent of the index)
 *    b) An internal randomized, unique 'id' to allow for differentiation between indices with the same order
 *    c) A vector of strings that allows these Indices to have tags, to allow for semantic differentiation
 *    d) An internal variable 'prime', (initialized to 0) that allows differentiation between copies of the same index
 *    e) An comparison operator == that checks if two Indices have the same order -> id -> tags -> prime
 *    f) A priming operator ' that primes an Index to its left (such that A' != A)
 *    g) cuTENSOR only cares about the key 'hash' and its value 'order' that it can add as a pair to an unordered
 *       map called extent, so we need to generate a unique 'hash' for each comparison value 
 *       NOTE: could just use the address of the Index
 * 
 * 2. Create a Tensor class that handles all the cuTENSOR jargon itself;
 *    a) A vector that holds the Indices of the Tensor, internally the vector should hold the hashes of the Indices
 *       that the Tensor is meant to have
 *    b) Another vector that holds the orders of these Indices
 *    c) An constructor that calculates the total size of the Tensor, so A_ij, where i and j have the extent 8, will
 *       have size of 8 x 8 = 64
 *    d) This size times the size of required half/float/double is the amount of memory we need to allocate
 *    e) We allocate this memory once on the host, and once on the device; we fill the values of the Tensor on the 
 *       host, and then cudaMemcpy to the device Tensor
 *    f) WARN: The biggest timesink of this, would be to write the logic for reshaping Tensors into matrices and back
 *       and having to communicate between these reshaped matrices and cuSOLVER
 *    g) 
 *       
 *
 *
 *
*/

static int i = 7; // no randomized hashing for now

class Index {
private:
	int order{0};
	int id;
	int prime{0};
public:
	Index(int ord){
		this->order = ord;
		this->id = i;
		i++;
	}

	Index(const Index& a){
		this->order = a.order;
		this->id = i;
		i++;
		this->prime = a.prime;
	}

	Index primed(Index a){
		Index b = a;
		this->prime++;
		this->id++;
		i+=2;
		return b;
	}

	int getOrder(){
		return this->order;	
	}

	int getID(){
		return this->id;
	}

	int getPrime(){
		return this->prime;
	}

};

class Tensor{
private:
	std::vector<Index*> indices;
public:
	Tensor(){
		this->indices[0] = nullptr;
	}
	Tensor(const std::vector<Index*> arr){
		for (int j = 0; j < arr.size(); j++){
			this->indices.push_back(arr[j]);	
		}
	}
	Tensor operator*(Tensor const& obj){
		
	}
};

int main(){
	Index a{3}, b{4}, c{5};
	std::vector<std::tuple<int, Index*>> keys;
	keys.push_back({a.getID(), &a});
	keys.push_back({b.getID(), &b});
	keys.push_back({c.getID(), &c});
	Tensor T(keys);
	std::cout << T.indices[7] << " " << T.indices[8] << " " << T.indices[9] << std::endl;
	// std::cout << *T.indices[7] << " " << *T.indices[8] << " " << *T.indices[9] << std::endl;
	
}
