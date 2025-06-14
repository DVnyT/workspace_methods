using MKL
using ITensors, ITensorMPS

using Plots

using Printf
using Statistics
using LsqFit

function DMRG()

        N = 100
        sites = siteinds("S=1/2", N)

        frac_values = Float64[]
        energy_values = Float64[]

        Sz_values = Float64[]
        Sx_values = Float64[]

        xi_values = Float64[]

        nsweeps = 30
        maxdim = 1000000
        cutoff = 1E-12

        iterations = 30
        start_frac = 0.9
        end_frac = 1.1
        step_frac = (end_frac - start_frac) / (iterations - 1)

        psi_i = random_mps(sites; linkdims=100)

        time_per_iteration = 0.0

        for i = 0:(iterations-1)
                @show i

                frac = start_frac + i * step_frac

                if i < 5
                        t_start = time()
                end

                @printf "--- Running DMRG for frac = %.5f (Iteration %d/%d) ---\n" frac (i + 1) iterations

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

                Sx = mean(expect(psi, "Sx"))
                Sz = mean(expect(psi, "Sz"))
                push!(Sx_values, Sx)
                push!(Sz_values, Sz)

                xi_val = NaN

                C = correlation_matrix(psi, "Sz", "Sz")

                ref = N ÷ 2

                avg_Sz_val = mean(expect(psi, "Sz"))

                distances_for_fit = Int[]
                correlations_for_fit = Float64[]

                for r = 1:(N-ref-1)
                        s = ref + r

                        connectedCorr = C[ref, s] - (avg_Sz_val^2)

                        if abs(connectedCorr) > 1E-16
                                push!(distances_for_fit, r)
                                push!(correlations_for_fit, abs(connectedCorr))
                        end
                end

                model_log_linear(r, p) = p[1] .+ p[2] .* r

                if length(distances_for_fit) > 5 && !isempty(correlations_for_fit)
                        p0 = [log(correlations_for_fit[1]), -0.05]

                        try
                                fit = curve_fit(model_log_linear, float.(distances_for_fit), log.(correlations_for_fit), p0)
                                param = coef(fit)
                                neg_inv_xi = param[2]

                                if neg_inv_xi < 0
                                        xi_val = -1 / neg_inv_xi
                                else
                                        xi_val = NaN
                                end
                        catch e
                                xi_val = NaN
                        end
                else
                        xi_val = NaN
                end
                push!(xi_values, xi_val)

                @printf "  Energy = %.10f ; h/J = %.5f ; Avg Sx = %.10f ; Avg Sz = %.10f ; Xi = %.5f\n" energy frac Sx Sz xi_val

                if i < 5
                        t_end = time()
                        time_per_iteration = ((time_per_iteration * i) + (t_end - t_start)) / (i + 1)

                        if i == 0
                                println("  Estimating time: Measuring first iteration...")
                        elseif i == 4
                                estimated_total_time = time_per_iteration * iterations
                                @printf "  Estimated time per iteration (avg of first %d): %.2f seconds\n" (i + 1) time_per_iteration
                                @printf "  Estimated total time for %d iterations: %.2f seconds (%.2f minutes)\n" iterations estimated_total_time (estimated_total_time / 60)
                                println("  (Note: This is an estimate; actual time may vary.)")
                        end
                end
        end

        println("\n--- Plotting Results ---")

        p1 = plot(frac_values, energy_values,
                xlabel="H/J (frac)",
                ylabel="Ground State Energy",
                title="DMRG Ground State Energy vs. H/J for N=$N",
                label="Energy",
                legend=:topright,
                marker=:circle,
                linewidth=2,
                linecolor=:blue,
                grid=true)

        p2 = plot(frac_values, Sz_values,
                xlabel="H/J (frac)",
                ylabel="⟨Sz⟩",
                title="Average ⟨Sz⟩ vs. H/J",
                label="⟨Sz⟩",
                legend=:topright,
                marker=:diamond,
                linewidth=2,
                linecolor=:red,
                grid=true)

        p3 = plot(frac_values, Sx_values,
                xlabel="H/J (frac)",
                ylabel="⟨Sx⟩",
                title="Average ⟨Sx⟩ vs. H/J",
                label="⟨Sx⟩",
                legend=:bottomright,
                marker=:utriangle,
                linewidth=2,
                linecolor=:green,
                grid=true)

        p4 = plot(frac_values, xi_values,
                xlabel="H/J (frac)",
                ylabel="Correlation Length (ξ)",
                title="Correlation Length vs. H/J for N=$N",
                label="ξ",
                legend=:topleft,
                marker=:circle,
                linewidth=2,
                linecolor=:orange,
                grid=true,
                yaxis=:log,
                ylims=(1, :auto),
        )

        combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1920, 1080))

        savefig(combined_plot, "crit_behaviour_Sz_xi.png")
        println("Combined plot saved to crit_behaviour_Sz_xi.png")

        return frac_values, energy_values, Sz_values, Sx_values, xi_values
end

DMRG()
