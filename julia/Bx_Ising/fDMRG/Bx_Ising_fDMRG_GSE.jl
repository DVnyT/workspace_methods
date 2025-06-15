using MKL

using ITensors, ITensorMPS

using Plots, Printf, Statistics

function DMRG()
        N = 100

        sites = siteinds("S=1/2", N)

        frac_values = Float64[]
        energy_values = Float64[]

        nsweeps = 5
        maxdim = 10000
        cutoff = 1E-12

        iterations = 30
        start_frac = 0.9
        end_frac = 1.1
        step_frac = (end_frac - start_frac) / (iterations - 1)

        psi_i = random_mps(sites; linkdims=100)

        for i = 0:(iterations-1)

                @show i

                frac = start_frac + i * step_frac

                os = OpSum()

                for i_site = 1:(N-1)
                        os += -4, "Sz", i_site, "Sz", (i_site + 1)
                end

                for j_site = 1:N
                        os += -2 * frac, "Sx", j_site
                end

                H = MPO(os, sites)

                energy, psi = dmrg(H, psi_i; nsweeps, maxdim, cutoff, outputlevel=0)

                psi_i = psi
                push!(frac_values, frac)
                push!(energy_values, energy)
        end


        println("\n--- Plotting Results ---")


        p1 = plot(frac_values, energy_values, xlabel="h/J (frac)", ylabel="Ground State Energy", title="DMRG GSE vs. h/J for N=$N", label="Energy", legend=:topright, marker=:circle, linewidth=2, linecolor=:blue, grid=true)


        savefig(p1, "Bx_Ising_fDMRG_GSE1.png")


end


DMRG()


