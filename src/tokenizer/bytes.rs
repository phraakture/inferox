//! GPT-2 byte-level BPE byte encoder/decoder tables.
//!
//! These map each byte 0..255 to a printable unicode codepoint so that raw
//! bytes can be represented as regular tokenizer vocabulary tokens.

use std::collections::HashMap;

/// Return the GPT-2 byte encoder table: index by byte value, get a `char`.
pub fn byte_encoder() -> Vec<char> {
    let mut n = 0u32;
    (0u8..=255u8)
        .map(|b| {
            let b_u32 = b as u32;
            if (0x21..=0x7E).contains(&b_u32)
                || (0xA1..=0xAC).contains(&b_u32)
                || (0xAE..=0xFF).contains(&b_u32)
            {
                char::from_u32(b_u32).unwrap()
            } else {
                let c = char::from_u32(0x0100 + n).unwrap();
                n += 1;
                c
            }
        })
        .collect()
}

/// Return the inverse mapping of `byte_encoder`.
pub fn byte_decoder() -> HashMap<char, u8> {
    byte_encoder()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as u8))
        .collect()
}
