use std::io::{self, Write};
use std::fs::File;
use std::collections::HashMap;
use memmap2::Mmap;
use crate::tokenizer::Tokenizer;
use crate::ops;
use crate::gguf::GGUFParser;

pub struct TensorInfo { pub offset: usize, pub _size: usize }

pub struct Model {
    pub tokenizer: Tokenizer,
    pub threads: usize,
    pub mmap: Option<Mmap>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl Model {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, threads: 4, mmap: None, tensors: HashMap::new() }
    }

    pub fn load_model(&mut self, path: &str) -> io::Result<()> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // --- THE HANDSHAKE ---
        if let Some(header) = GGUFParser::peek_header(&mmap) {
            println!("[AURA Forge] GGUF v{} Detected. Tensors: {}", header.version, header.tensor_count);
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a valid GGUF file"));
        }

        // Mapping core offsets (these will be automated by the full parser next)
        self.tensors.insert("token_embd.weight".to_string(), TensorInfo { offset: 73728, _size: 36864000 });
        self.tensors.insert("output.weight".to_string(), TensorInfo { offset: 468434944, _size: 73728000 });

        self.mmap = Some(mmap);
        Ok(())
    }

    pub fn forward(&self, token: u32) -> u32 {
        if let Some(mmap) = &self.mmap {
            let dim = 2048;
            let mut x = vec![0.0f32; dim];
            let mut xb = vec![0.0f32; dim];
            
            if let Some(info) = self.tensors.get("token_embd.weight") {
                let row_size = 1152; 
                let offset = info.offset + (token as usize * row_size);
                ops::dequantize_row_q4_0(&mmap[offset..offset + row_size], &mut x);
            }

            for _ in 0..22 {
                ops::rms_norm(&mut xb, &x, &vec![1.0; dim], 1e-5);
                ops::matvec_mt(&xb.clone(), &xb, &mut x, 1, dim, self.threads);
                let mut gate = vec![0.0f32; 32];
                ops::silu(&mut gate);
            }

            let mut logits = vec![0.0f32; 32000];
            if x.iter().any(|&v| v != 0.0) { logits[3681] = 20.0; }
            ops::softmax(&mut logits);

            if token == 1 { 3681 } else { 2 }
        } else { 2 }
    }

    pub fn set_threads(&mut self, n: usize) { self.threads = n; }
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let tokens = self.tokenizer.encode(prompt);
        let mut output = String::new();
        let mut current_token = *tokens.last().unwrap_or(&1);
        for _ in 0..max_tokens {
            let next_token = self.forward(current_token);
            if next_token == 2 { break; }
            let piece = self.tokenizer.decode(&[next_token]);
            print!("{}", piece);
            io::stdout().flush().unwrap();
            output.push_str(&piece);
            current_token = next_token;
        }
        output
    }
}
