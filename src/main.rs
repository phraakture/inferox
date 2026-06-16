use std::env;
use std::process;

use inferox::gguf::{GgufFile, TensorInfo, Weights};

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "model.gguf".to_string());

    let weights = match Weights::open(&path) {
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
    print_layout(file, &path);
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
