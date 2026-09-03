Hi all,

In 2023, I was one of the guys co-authoring GGUF. Over the years, it was a great pleasure to see how it grew and adopted by community. It's a very cute, handsome or beautiful format, at your option. However, it requires implementing the execution graph of each model architecture from scratch in C++, so you need to debug many things all together: reference implementation in transformers or another library, trace the hell of over-abstracted subclassing and mixin dependencies, compilation of the C++ code over and over again.



I'm always an admirer of ML compilers such as XLA and executorch, and the engineering behind their step-by-step optimization to lower high-level ML ops to fused kernels at the hardware level. 



Combining both, I implemented ggmlc, a multi-framework neural network compiler lowering PyTorch, JAX, Flax, and Keras models to portable, high-performance GGML execution. 



ggmlc_t4_linkedin_card.png
`ggmlc` eliminates the overhead of writing brittle, hand-crafted C++ inference code for each new model architecture by treating neural networks as semantic tensor programs:



1. Zero Hand-Written C++ Glue: Ingests models directly from PyTorch (`torch.export`) and JAX/Flax (`jaxpr`), translates them into strongly-typed Canonical IR, and optimizes them automatically.

2. Standard GGUF v3 Containers: Serializes graphs, dynamic shapes, and quantized weights into standard `.gguf` binaries — no proprietary file formats or runtime lock-in.

3. CPU & NVIDIA CUDA GPU Backends: Run models directly on CPU or NVIDIA GPUs with zero-copy VRAM buffer transfers, device placement (`device="cuda"`, `device="cpu"`, `device="auto"`), and native CUDA fused ops.

4. Standalone Human-Readable C++ Code Generation: Alternatively emits self-contained C++ header files (`<Model>.h`), native entry points (`ggmlc_main.cpp`), and `CMakeLists.txt` for direct embedding into native applications with CPU/CUDA backend support.

5. 100% Golden-Truth Numerical Parity: Automated differential numerical testing guarantees exact mathematical parity (> 0.99999 cosine similarity) against PyTorch and JAX reference runs on both CPU and GPU.

6. High-Performance Python Binding (`nanobind`): Zero-copy NumPy buffer evaluation with multi-threaded CPU execution and streaming serialization.



To my knowledge, this is the first direct integration of JAX and GGML/GGUF ecosystems. For most of the golden architectures in JAX, I used Keras3 with a JAX backend and Keras Hub for the model weights. A multi-backend framework also allowed me to cross-validate correctness of lowering and improve pattern-matcher for operation fusion for JAX. A more detailed article on this will be coming soon.



If you have any questions, comments, or anything else to discuss, feel free to reach out to me directly.



Here are some links:

If you want to give a start or like:

ggmlc repo: https://github.com/monatis/ggmlc

LinkedIn post: https://lnkd.in/p/dYZnbMfR

Demo notebook on Colab: https://colab.research.google.com/drive/1jD5Pr4ObD9CGoRoC7_LQmAGvh0AZZ6KW?usp=sharing


