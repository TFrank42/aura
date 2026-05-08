mod gguf;
mod ops;
mod model;
mod tokenizer;

use std::env;
use crate::model::Model;
use crate::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: aura-probe <model_path> <prompt>");
        return;
    }

    let model_path = &args[1];
    let prompt = &args[2];

    println!("[AURA Forge] Mapping weights from: {}", model_path);
    let mut tokenizer = Tokenizer::new();
    tokenizer.load("tokenizer.json"); 

    let mut model = Model::new(tokenizer);
    model.set_threads(6); 

    if let Err(e) = model.load_model(model_path) {
        println!("Failed to load model: {}", e);
        return;
    }

    println!("[AURA Scout] 8-Core DNA detected. Engaging 6 threads.");
    println!("Prompt: {}", prompt);
    print!("Generating... ");
    
    model.generate(prompt, 32);
    println!("\n[Inference Complete]");
}
