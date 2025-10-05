//! Example: Batch process multiple audio files

use charon_audio::{utils, Separator, SeparatorConfig};
use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_dir> [output_dir] [model_path]", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} audio_files/ separated/ model.onnx", args[0]);
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = args.get(2).map(|s| s.as_str()).unwrap_or("batch_output");
    let model_path = args.get(3).map(|s| s.as_str()).unwrap_or("model.onnx");

    println!("Charon Batch Audio Separator");
    println!("============================");
    println!("Input dir:  {input_dir}");
    println!("Output dir: {output_dir}");
    println!("Model:      {model_path}");
    println!();

    // Find all audio files
    println!("Scanning for audio files...");
    let audio_files = utils::find_audio_files(input_dir, true)?;
    println!("Found {} audio files", audio_files.len());
    println!();

    if audio_files.is_empty() {
        println!("No audio files found in {input_dir}");
        return Ok(());
    }

    // Create separator
    let config = SeparatorConfig::onnx(model_path)
        .with_shifts(1)
        .with_segment_length(10.0);

    println!("Loading model...");
    let separator = Separator::new(config)?;

    // Process all files
    println!("Processing files...");
    separator.separate_batch(&audio_files, output_dir)?;

    println!("\n✓ Batch processing complete!");
    println!("Output saved to: {output_dir}");

    Ok(())
}
