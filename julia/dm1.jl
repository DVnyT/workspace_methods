using ITensors, ITensorMPS

let
    N = 100
    sites = siteinds("S=1/2", N)
    
    J = 1
    H = 1
    
    os = OpSum()
    
    for i = 1: (N - 1)
        os += -4J, "Sz", i, "Sz", (i + 1)
    end
    for j = 1: N
        os += -2H, "Sx", j
    end
    
    H = MPO(os, sites)
    
    nsweeps = 5
    maxdim = [10, 20, 100, 200, 1000]
    cutoff = 1E-12
    prev_states = MPS[]        # will hold psi1, ps  i2, …
    energies    = Float64[]    # will hold E1, E2  , …
    


    for k in 1:50
      psi0 = randomMPS(sites; linkdims=100)
    
      E, psi = k == 1 ?
        dmrg(H, psi0; nsweeps, maxdim, cutoff) :
        dmrg(H, prev_states, psi0; nsweeps, maxdim, cutoff)
    
      push!(energies, E)
      push!(prev_states, psi)
    end
    @show energies
    
end 
