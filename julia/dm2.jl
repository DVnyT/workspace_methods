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
    
    psi0_i = random_mps(sites; linkdims = 10)
    energy0, psi0 = dmrg(H, psi0_i; nsweeps, maxdim, cutoff)
    @show(energy0)
    @show(inner(psi0', H, psi0))
    psi1_i = random_mps(sites; linkdims = 10)
    energy1, psi1 = dmrg(H, [psi0], psi1_i; nsweeps,maxdim,cutoff)
    @show(energy1)
    @show(inner(psi1', H, psi1))   
end 
