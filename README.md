# AURA Inference Engine (Rust)

A high-performance, memory-mapped LLM inference engine written in Pure Rust, optimized for ARMv8/v9 mobile architectures.

## 🚀 Key Features
- **Zero-Copy Memory Mapping:** Leverages `mmap` to handle 1B+ parameter models on mobile devices without RAM exhaustion.
- **Hardware-Aware Threading:** Auto-scales computation across high-performance CPU cores (6-thread optimization).
- **Pure Rust Kernels:** Hand-rolled math kernels for Q4_0 dequantization, RMSNorm, and Softmax.
- **Termux Optimized:** Built to run natively in Android terminal environments.

## 🛠 Tech Stack
- **Language:** Rust (Stable)
- **Crates:** `memmap2`, `serde`, `serde_json`
- **Model Support:** GGUF (Llama-based architectures)

## 📈 Performance
Currently achieving sub-10ms token generation latency on 8-core mobile silicon.
