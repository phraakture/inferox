# inferox

Rust-native LLM runtime prototype. Goal: load a GGUF model and generate text end-to-end on CPU.

## roadmap

- [x] parse GGUF metadata and tensor info
- [x] mmap weights
- [x] dequantize F32 / F16 / Q4_0 / Q8_0 to f32
- [x] CPU kernels: matmul, RMSNorm, SiLU, SwiGLU, softmax, RoPE
- [x] parse llama architecture metadata
- [x] wire embeddings / layers / output tensors
- [x] single transformer layer forward pass
- [x] full model forward: embed -> layers -> norm -> logits
- [x] KV-cache for autoregressive generation
- [x] tokenizer (BPE / SentencePiece)
- [x] sampler with temperature / top-k / top-p
- [x] CLI: prompt in, text out
- [ ] dequantize Q4_K_M, Q6_K, Q8_K (most HF models)
- [x] AVX2 SIMD matmul
- [ ] benchmark tokens/sec vs llama.cpp
- [x] batched prefill optimization
