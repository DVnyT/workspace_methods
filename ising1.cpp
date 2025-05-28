# include <iostream>
# include <vector>
# include <map>
# include <tuple>
/*  
 *  We want to do some manual tensor contractions for an Transverse-field Ising model, and see how the sweeps are 
 *  setup. Let's just follow the basic structure from ITensors as follows
 *  1. Create a class Index that has d legs
 *  2. Create a class Tensor that can contract indices via an operator overload *
 *  3. Create an MPS, MPO, and look up eigensolvers
 *  4. Do the DMRG sweeps
*/

static int i = 7;

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
