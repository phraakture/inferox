# inferox

Rust-native CPU inference for Llama-style GGUF models. No bindings, no GPU, just the bits needed to load a model and generate text.

## what it does

- Parses GGUF v2/v3 files
- Memory-maps weights instead of copying them
- Dequantizes F32, F16, Q4_0, Q8_0, Q4_K, Q6_K, Q8_K
- Runs Llama architecture on CPU
- KV-cache autoregressive generation
- BPE tokenizers from Hugging Face `tokenizer.json`
- Temperature / top-k / top-p sampling
- Batched prefill + single-token decode
- AVX2+FMA matmul on x86_64

## build

```bash
cargo build --release
```

## run

Inspect a GGUF file:

```bash
cargo run --release -- model.gguf
```

Generate text:

```bash
cargo run --release -- model.gguf tokenizer.json "what is rust" 64 0.8
```

Arguments: `model.gguf tokenizer.json prompt max_tokens temperature`. Temperature `0` is greedy argmax.

## as a library

```rust
use inferox::model::{Model, KvCache, DecodeBuffers};
use inferox::tokenizer::BpeTokenizer;
use inferox::sampler::Sampler;

let model = Model::open("model.gguf")?;
let tokenizer = BpeTokenizer::from_file("tokenizer.json")?;

let prompt = tokenizer.encode("hello");
let mut kv = KvCache::new(model.config.n_layers, model.config.context_length, &model.config);
let mut buf = DecodeBuffers::new(model.config.context_length, &model.config);
let sampler = Sampler { temperature: 0.8, top_k: 0, top_p: 1.0 };

let mut rng = rand::thread_rng();
let mut generated = Vec::new();

for &t in &prompt {
    model.decode_step(t, &mut kv, &mut buf)?;
}

for _ in 0..64 {
    let next = sampler.sample(&buf.logits, &mut rng);
    generated.push(next);
    model.decode_step(next, &mut kv, &mut buf)?;
}

println!("{}", tokenizer.decode(&generated));
```

## project layout

```
src/
  gguf/       parser, mmap, dequantization
  model/      config, layer wiring, forward pass, kv cache
  ops.rs      CPU kernels (matmul, norm, rope, softmax, swiglu)
  sampler.rs  temperature / top-k / top-p
  tokenizer/  BPE tokenizer.json loader
  main.rs     CLI
```

## status

It runs real GGUF models end-to-end on CPU but is not optimized to match llama.cpp speed. The main gaps are quantized matmul, multi-threading, and more architectures.

## license

MIT
