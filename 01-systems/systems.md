# 01 · Systems

The software that turns hardware into fast tensor math: GPU programming, kernels, compilers and profiling.

## Topics

| Topic | What it covers | Status |
|---|---|---|
| CUDA programming model | Grids, blocks, threads, memory coalescing, shared memory, occupancy | planned |
| Kernel languages | Triton, CUTLASS/CuTe, ThunderKittens, TileLang, ROCm/HIP, Pallas | planned |
| Key kernels | GEMM, softmax, RMSNorm fusion, FlashAttention, MoE grouped GEMM | planned |
| Compilers & graphs | `torch.compile`, XLA, MLIR, CUDA graphs, mega-kernels | planned |
| Profiling | Nsight, PyTorch profiler, roofline analysis | planned |
| AI-written kernels | LLM-driven kernel search and benchmarks | planned |

## Start here

- GPU MODE lectures and reference kernels (P)
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel* (P)
- Triton tutorials; Karpathy's `llm.c` (P)
