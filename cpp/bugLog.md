1. .reserve() does not actually increase the size of a vector (and only allocs memory). This in turn means that 
using `vec[i] = val`; is actually illegal, and thus vec is never updated as thought!

