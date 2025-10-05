//! Example: Separate audio file into stems

use charon_audio::{Separator, SeparatorConfig};
use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_audio> [output_dir] [model_path]", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} song.mp3 output/ model.onnx", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_dir = args.get(2).map(|s| s.as_str()).unwrap_or("separated");
    let model_path = args.get(3).map(|s| s.as_str()).unwrap_or("model.onnx");

    println!("Charon Audio Separator");
    println!("======================");
    println!("Input:  {input_path}");
    println!("Output: {output_dir}");
    println!("Model:  {model_path}");
    println!();

    // Create configuration
    let config = SeparatorConfig::onnx(model_path)
        .with_shifts(1)
        .with_segment_length(10.0)
        .with_progress(true);

    println!("Loading model...");
    let separator = Separator::new(config)?;

    println!("Processing audio...");
    let stems = separator.separate_file(input_path)?;

    println!("\nSeparated stems:");
    for name in stems.list() {
        println!("  - {name}");
    }

    println!("\nSaving stems to {output_dir}...");
    stems.save_all(output_dir)?;

    println!("\n✓ Done!");
    Ok(())
}
