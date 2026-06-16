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
- [ ] full model forward: embed -> layers -> norm -> logits
- [ ] KV-cache for autoregressive generation
- [ ] tokenizer (BPE / SentencePiece)
- [ ] sampler (greedy / top-k / top-p)
- [ ] CLI: prompt in, text out
- [ ] more quant types: Q5_0, Q8_K, IQ quants
