//! Example: Separate audio file into stems
//!
//! Usage: separate <input_audio> [output_dir] [model_path] [--shifts N]
//!        [--format wav16|wav24|wav32|flac16|flac24] [--max-speed] [--split]
//!        [--ep cpu|coreml|auto]

use charon_audio::{BitDepth, Separator, SeparatorConfig, StemFormat};
use charon_audio::{ExecutionProvider, OnnxOptions};
use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut positional = Vec::new();
    let mut shifts = 1usize;
    let mut format = StemFormat::default();
    let mut max_speed = false;
    let mut split = false;
    let mut ep = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--shifts" => shifts = args.next().expect("--shifts N").parse()?,
            "--format" => {
                format = match args.next().expect("--format NAME").as_str() {
                    "wav16" => StemFormat::Wav(BitDepth::Int16),
                    "wav24" => StemFormat::Wav(BitDepth::Int24),
                    "wav32" => StemFormat::Wav(BitDepth::Float32),
                    "flac16" => StemFormat::Flac(BitDepth::Int16),
                    "flac24" => StemFormat::Flac(BitDepth::Int24),
                    other => anyhow::bail!("unknown format {other}"),
                }
            }
            "--max-speed" => max_speed = true,
            "--split" => split = true,
            "--ep" => {
                ep = Some(match args.next().expect("--ep NAME").as_str() {
                    "cpu" => ExecutionProvider::Cpu,
                    "coreml" => ExecutionProvider::CoreMl,
                    _ => ExecutionProvider::Auto,
                })
            }
            _ => positional.push(arg),
        }
    }
    if positional.is_empty() {
        eprintln!("Usage: separate <input_audio> [output_dir] [model_path] [--shifts N] [--format wav16|wav24|wav32|flac16|flac24] [--max-speed] [--split] [--ep cpu|coreml|auto]");
        std::process::exit(1);
    }

    let input_path = &positional[0];
    let output_dir = positional.get(1).map(|s| s.as_str()).unwrap_or("separated");
    let model_path = positional
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("htdemucs.onnx");

    println!("Charon Audio Separator");
    println!("======================");
    println!("Input:  {input_path}");
    println!("Output: {output_dir}");
    println!("Model:  {model_path}");
    println!("Shifts: {shifts}");
    println!();

    // HTDemucs ONNX export: 4 stems, fixed 7.8 s segments
    let mut config = if split {
        SeparatorConfig::htdemucs_split(model_path)
    } else {
        SeparatorConfig::htdemucs(model_path)
    }
    .with_shifts(shifts)
    .with_progress(true);
    if max_speed {
        config.model.onnx = OnnxOptions::max_speed();
    }
    if let Some(ep) = ep {
        config.model.onnx.execution_provider = ep;
    }

    println!("Loading model...");
    let separator = Separator::new(config)?;

    println!("Processing audio...");
    let stems = separator.separate_file(input_path)?;

    println!("\nSeparated stems:");
    for name in stems.list() {
        println!("  - {name}");
    }

    println!("\nSaving stems to {output_dir}...");
    stems.save_all_as(output_dir, format)?;

    println!("\nDone.");
    Ok(())
}
