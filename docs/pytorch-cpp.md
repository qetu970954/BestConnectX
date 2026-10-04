# PyTorch in C++: research and selected route

[繁體中文](pytorch-cpp.zh-TW.md)

**Selected route: direct LibTorch tensor operations in C++. Python/PyTorch keeps model learning. Windows CPU/CUDA use is now checked.** See the [native guide](native-engine.md) and [local results](migration-validation.md).

## Does C++ make PyTorch faster?

Not by itself. PyTorch's Python interface already calls compiled C++/CUDA math. The official C++ guide warns that Python is not always slower. C++ helps when Python work between operations or low-delay integration matters. [1]

For this engine, the whole playing path and inference now stay in C++. This removes Python board/leaf creation and Python model calls from the search loop. CPU/GPU transfers and GPU launch costs remain.

Local paired search tests show a gain under the tested settings. They do not isolate the language of the model call as the only cause, or prove more strength per training hour.

## Terms and evidence

- **Inference:** predict move scores and a board value without changing weights.
- **LibTorch:** PyTorch's C++ headers and libraries for tensors, models, gradients, CPU, and GPU. [1][2]
- **Dispatch cost:** CPU work that requests a model operation before its math runs.
- **CUDA graph:** saved GPU operations replayed through one launch request. It can avoid Python, C++, and driver dispatch work, but needs stable memory and compatible shapes/control flow. [3]
- **AOTInductor:** a tool that compiles an exported PyTorch model for Python/C++ use. The 2.11 guide shows `torch.export` and a C++ package loader. [5]

## Why this route

`engine/inference.cpp` implements the same residual, pooled, and attention architectures as `engine/network.py`, using LibTorch tensor operations. Python passes updated weights and running buffers at safe points. Both paths share architecture/width/depth/board settings and are checked against the same inputs.

The build uses the installed PyTorch wheel. On this Windows machine, MSVC Release-style linking and CPU/CUDA inference work. No TorchScript/export loader, CMake, or separate CUDA SDK is needed for this route. This does not establish Windows CUDA support for AOTInductor or `torch.compile`.

TorchScript is deprecated in the 2.11 docs. [6] Export/AOTInductor remains a possible later deployment path. It is not required while Python still owns learning and can pass weights to the native model directly.

## Other options kept out of this release

- Python/PyTorch inference can be a smaller port, but it is not the user's selected scope.
- CUDA graphs can reduce launch work in either language. Dynamic self-play batches need a graph-safe design. [3]
- `torch.compile(mode="reduce-overhead")` uses CUDA graphs where possible and can need more workspace. It does not promise a gain. [4]
- The Windows compiler tutorial establishes CPU/XPU setup, not this Windows/NVIDIA CUDA compiler route. [7]

Add these only after measuring a remaining need. No performance claim is based on those untested routes.

## Checks and limits

CPU and CUDA checks compare model outputs with the learner. After real updates, they reload weights and running buffers and compare again. Full games and temporary train/resume checks cover 3×3, 15×15, and 19×19 in that order.

Paired search timings use the same weights, rules, batch, simulations, and FP32 mode on both sides. This is a test control, not an exact-repeat promise. Warm search medians exclude build, model loading, and warmup. The full GPU-process wall-time budget includes startup and warmup. Native inference timing includes input/output transfers and waits; it is not pure GPU kernel time.

The agent's permission remains five minutes of total temporary GPU-process wall time, with no persistent training. Long strength comparisons are user-run work. File I/O is measured separately and is not fixed by moving inference to C++.

## Official sources

1. [PyTorch C++ interface: scope and performance warning](https://docs.pytorch.org/cppdocs/frontend.html)
2. [Install and link LibTorch](https://docs.pytorch.org/cppdocs/installing.html)
3. [PyTorch: CUDA graphs and their limits](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
4. [PyTorch 2.11: `torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
5. [PyTorch 2.11: AOTInductor and C++ inference](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
6. [PyTorch 2.11: TorchScript deprecation notice](https://docs.pytorch.org/docs/2.11/notes/cpu_threading_torchscript_inference.html)
7. [PyTorch: compiler setup on Windows CPU/XPU](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
