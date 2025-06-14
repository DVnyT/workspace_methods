using ITensor

### Index()

# 1) Use Index to create an index of (param) dimensions, naturally has an internal unique hash to differentiate between#    2 indices of the same dim  
i = Index(3)
@show i  # This shows (dim=3|id=xxx)  
j = Index(3)
@show i == j  # False
hasid(i, id(j))  # False
dim(i)

# We can create a copy with a different id
cpy = sim(i)

# 2) Indices can have "tags" which add another layer of differentiation
k = i
k = settags(k, "tag1")
@show k == i  # False
tags(k)       
hastags(k, "tag1") # True
addtags(k, "tag2,tag3")
replacetags(k, "tag2" => "tag3")

# An index can be primed, which is internally a special integer tag
l = i'
@show l == i  # False
m = (prime(i), 2)'  # 1 by default
@show plev(m) - plev(i)  # plev() only tracks the integer, not who an index is primed against!

# Other methods!
p = adjoint(i)
p == i'  # True
p = i^3
p == i'  # False
p = noprime(p)
p == i  # True

# 3) A direction can be defined!
dir(i)
# and reversed,
dag(i)

# 4) Iterators can be created for manual traversal
eachval(i)  # TODO
eachindval(i)  # TODO

### ITensor()

# 1) Create a tensor with ITensor()
A = ITensor(ComplexF64, i,j)
B = ITensor(3.0, j,i)  # All values are 3.0
showinds(B)
A[i=>3, j=>2] = 7.0 + 0.2im  # Set values

C = randomITensor(i, j)
D = randomITensor(i, j, k)
@show C
@show storage(C)

E = onehot(i=>2, j=>3)  # Makes a tensor with [i=>2, j=>3] being 1 and the rest 0

F = diag_itensor(i, j)  # Diagonal! 

G = delta(i, j)  # All diagonal elements set to 1, rest 0
@show D * G

# A delta tensor, contracted with a tensor it shares a common index with, will REPLACE that index by the rest of the 
# delta tensor! 
G = delta(i,l,k) * A # A_ij
@show G # should be A_jlk
hasind(G,j)*hasind(G,l)*hasind(G,k) == 1 # can straight up use the unicode δ lol

# A combiner is kind of the opposite, when acted on a tensor it shares indices with, it will combine all its shared 
# indices
H = randomITensor(i,j,k)
Com = combiner(i,j)
ComH = Com * H # returns a tensor with the 2 indices k and [i,j] 
@show combinedind(Com)
@show commonind(Com, ComH)
# Can conjugate the combiner to reverse this operation as dag(Com) * ComH == H

# Can have empty tensors
m = Index(20)
V = [randomITensor(m), randomITensor(m)]
X = ITensor() # no arguments
for A in V, # julia loop!
	X += A
end
# does *= work for contractions?!
# can Add (between two tensors with the same indices)

# //TODO Some Julia array hacks! 

# 2) Prime an index with prime()
Prime(B,i)
@show A * B # this will NOT return a scalar!
hasind(A * B, i)*hasind(A * B, i') # should be = 1; obviously don't compute the same contraction twice IRL

# 3) Index Manipulation

# 4) Math
@show A + C
# and contract
@show A * D
#    As before, we can add and contract, we can additionally get the hermitian conjugate
dag(A) * A

### Decomposition
# 1) qr()
T = randomITensor(i,j,k)
Q,R = qr(T, (i,j)) # The passed (i,j) implicitly combines (i,j) and takes them to be the row index!
commonind(Q,R) # passes the new index created

# 2) svd()
U, S, V = svd(T, (j,i)) # again the (j,i) are row indices, leaving k to be the column index
# Truncated decompositions are specified by some parameters
U', S', V' = svd(T, (j,i); cutoff = 1E-10, maxdim = 1000) # maxdim is D, cutoff must be for the frobenius norm         difference

### MPOs
function heisenberg_mpo(N)
# Make N S=1/2 spin indices
sites = siteinds("S=1/2",N) # interesting function!
# Input the operator terms
os = OpSum()
for i=1:N-1
os += "Sz",i,"Sz",i+1
os += 1/2,"S+",i,"S-",i+1
os += 1/2,"S-",i,"S+",i+1
end
# Convert these terms to an MPO
H = MPO(os,sites)
return H
end
H = heisenberg_mpo(100) # sure seems contrived...


---------
# DMRG
---------
# We input a Hamiltonian H and an initial guess for psi(i) for the ground state psi
# We start with the Hamiltonian above
state = [isodd(n) ? "Up":"Dn" for n = 1:N] # ??? oh a ternary expression?
psi0_i = MPS(sites, state) # honestly seems like a cheatcode
# We do 10 sweeps
# Increasing the bond dimension D at each stage
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 100, 200, 400, 800)
setcutoff!(sweeps, 1E-8)                        # both of our SVD parameters from before!
energy, psi0 = dmrg(H, psi0_i, sweeps)

# For hamiltonians defined as a sum we can run as
energy, psi0 = dmrg([H1,H2,H3], psi0_i, sweeps)

# For excited states, we add the constraint that the new state to be found is orthogonal to the previous ones,
energy2, psi2 = dmrg(H, [psi0,psi1], psi2_i, sweeps) # note that we also need to pass a guess for the excited state


## MPS
# An MPS is a factorization of a tensor psi
# To get the A at the jth index,
Aj = psi[j]
# and updated as,
psi[j] = new_Aj

# we can have functions
avgSz = expect(psi, "Sz") # we compute the expectation value of Sz for every site and return an array

C = correlation_matrix(psi, "S+", "S-")

# IDK if the next one really does a mixed canonical MPS or not, but it does orthogonalize the MPS about a center j
orthogonalize!(psi, j)

truncate!(psi; maxdim = 500, cutoff = 1E-8)

# Can add MPS with truncation as
eta = add(psi, phi; cutoff = 1E-10)

# and can approximately multiply as
Wpsi = contract(W, psi; maxdim = 50)
# and even MPOs
RW = contract(R, W; cutoff = 1E-9) # have additional parameters like method = "naive"

