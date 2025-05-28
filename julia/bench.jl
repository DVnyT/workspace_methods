#!/usr/bin/env julia
# benchmark_itensors.jl

using ITensors           # core library
using BenchmarkTools     # for precise timings
using CUDA               # CUDA.jl
using cuTENSOR           # cuTENSOR.jl extension

println("\n=== Warming up JIT & GPU context ===")
let
  I, J, K = Index(3), Index(5), Index(8)
  T1, T2, T3 = randomITensor(I, J), randomITensor(J, K), randomITensor(I, J, K)
  # CPU path
  _ = T1 * T2 * T3
  # GPU path
  T1g, T2g, T3g = cu(T1), cu(T2), cu(T3)
  _ = T1g * T2g * T3g
  CUDA.synchronize()
end
println(" Warm‐up complete.")

# -----------------------------------------------------------------------------
# 3) Microbenchmark on tiny 3×5×8 contraction
# -----------------------------------------------------------------------------
println("\n=== Microbenchmark: 3×5×8 contraction ===")
let
  I, J, K = Index(3), Index(5), Index(8)
  T1, T2, T3 = randomITensor(I, J), randomITensor(J, K), randomITensor(I, J, K)
  # CPU
  @btime $T1 * $T2 * $T3
  @btime $T1 * $T2 * $T3
  # GPU
  T1g, T2g, T3g = cu(T1), cu(T2), cu(T3)
  @btime $T1g * $T2g * $T3g
  CUDA.synchronize()
end

# -----------------------------------------------------------------------------
# 4) Scale‐up loop: compare CPU vs GPU as D grows
# -----------------------------------------------------------------------------
println("\n=== Scaling test: inner loop A[i,j] * B[j,k] ===")
for D in (256, 512, 1024, 20000)
  println("\n– Bond dim D = $D –")
  idx1, idx2 = Index(D), Index(D)
  A = randomITensor(idx1, idx2)
  B = randomITensor(idx2, idx1)
  A_g, B_g = cu(A), cu(B)
  # warm up
  _ = A * B
  _ = A_g * B_g; CUDA.synchronize()
  # CPU benchmark
  cpu_b = @benchmark $A * $B
  println(" CPU: D=$D → ", minimum(cpu_b.times))
  # GPU benchmark
  gpu_b = @benchmark begin
    tmp = $A_g * $B_g
    CUDA.synchronize()
  end
  println(" GPU: D=$D → ", minimum(gpu_b.times))
end

println("\nBenchmark script complete.")


