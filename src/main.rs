use std::env;
use std::process;

use inferox::gguf::{GgufFile, TensorInfo, Weights};
use inferox::model::{DecodeBuffers, KvCache, Model};
use inferox::sampler::Sampler;
use inferox::tokenizer::{BpeTokenizer, Tokenizer};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: {} <model.gguf> [tokenizer.json] [prompt] [max_tokens] [temperature]", args[0]);
        process::exit(1);
    }

    let model_path = &args[1];

    if args.len() < 4 {
        inspect(model_path);
        return;
    }

    let tokenizer_path = &args[2];
    let prompt = &args[3];
    let max_tokens: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let temperature: f32 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    generate(model_path, tokenizer_path, prompt, max_tokens, temperature);
}

fn generate(model_path: &str, tokenizer_path: &str, prompt: &str, max_tokens: usize, temperature: f32) {
    let model = match Model::open(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to load model '{}': {e}", model_path);
            process::exit(1);
        }
    };

    let tokenizer = match BpeTokenizer::from_file(tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to load tokenizer '{}': {e}", tokenizer_path);
            process::exit(1);
        }
    };

    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq_len = model.config.context_length;
    let mut kv_cache = KvCache::new(model.config.n_layers, max_seq_len, &model.config);
    let mut buf = DecodeBuffers::new(max_seq_len, &model.config);
    let mut rng = rand::thread_rng();
    let sampler = Sampler {
        temperature,
        top_k: 0,
        top_p: 1.0,
    };

    let mut generated_tokens = Vec::new();

    for &token in &prompt_tokens {
        if let Err(e) = model.decode_step(token, &mut kv_cache, &mut buf) {
            eprintln!("error: decode failed: {e}");
            process::exit(1);
        }
    }

    for _ in 0..max_tokens {
        let next_token = sampler.sample(&buf.logits, &mut rng);
        generated_tokens.push(next_token);
        if let Err(e) = model.decode_step(next_token, &mut kv_cache, &mut buf) {
            eprintln!("error: decode failed: {e}");
            process::exit(1);
        }
    }

    print!("{}", tokenizer.decode(&generated_tokens));
}

fn inspect(path: &str) {
    let weights = match Weights::open(path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to open '{}': {e}", path);
            process::exit(1);
        }
    };

    let file = weights.file();
    print_header(file);
    print_metadata(file);
    print_tensors(file);
    print_layout(file, path);
    print_tensor_bytes(&weights, file);
}

fn print_header(file: &GgufFile) {
    section("Header");
    println!("  version:     {}", file.header.version);
    println!("  n_tensors:   {}", file.header.n_tensors);
    println!("  n_kv:        {}", file.header.n_kv);
}

fn print_metadata(file: &GgufFile) {
    section(&format!("Metadata ({} entries)", file.metadata.len()));
    for (i, meta) in file.metadata.iter().enumerate() {
        println!(
            "  [{:3}] {:50} ({}) => {}",
            i,
            meta.key,
            meta.value_type(),
            meta.value
        );
    }
}

fn print_tensors(file: &GgufFile) {
    section(&format!("Tensors ({} tensors)", file.tensors.len()));
    for (i, tensor) in file.tensors.iter().enumerate() {
        print_tensor(i, tensor);
    }
}

fn print_tensor(index: usize, tensor: &TensorInfo) {
    let shape = tensor
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" x ");

    println!("  [{:3}] {}", index, tensor.name);
    println!("        type:    {} (id={})", tensor.ty, tensor.ty as u32);
    println!("        shape:   [{}]", shape);
    println!("        elems:   {}", tensor.n_elements());
    println!("        offset:  {} bytes", tensor.offset);
    match tensor.byte_size() {
        Some(size) => println!("        size:    {} bytes", size),
        None => println!("        size:    <unknown type size>"),
    }
}

fn print_layout(file: &GgufFile, path: &str) {
    section("Layout");
    println!("  file:                 {}", path);
    println!(
        "  tensor data offset:   {} bytes",
        file.tensor_data_offset
    );
    match file.total_tensor_data_size() {
        Some(size) => println!("  total tensor bytes:   {} bytes", size),
        None => println!("  total tensor bytes:   <unknown>"),
    }
}

fn print_tensor_bytes(weights: &Weights, file: &GgufFile) {
    section("Raw tensor bytes (mmap)");
    if file.tensors.is_empty() {
        println!("  (no tensors)");
        return;
    }

    for (name, bytes) in weights.tensors().take(5) {
        match bytes {
            Ok(b) => println!("  {:40} -> {} bytes", name, b.len()),
            Err(e) => println!("  {:40} -> error: {e}", name),
        }
    }
    let remaining = file.tensors.len().saturating_sub(5);
    if remaining > 0 {
        println!("  ... and {remaining} more tensor(s)");
    }
}

fn section(title: &str) {
    const WIDTH: usize = 72;
    println!("{}", "-".repeat(WIDTH));
    println!("{}", title);
    println!("{}", "-".repeat(WIDTH));
}
