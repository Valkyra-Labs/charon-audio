//! Example: Real-time audio separation (conceptual)

use charon_audio::{AudioBuffer, Separator, SeparatorConfig};
use ndarray::Array2;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("Charon Real-time Audio Separator");
    println!("=================================");
    println!();

    // Create separator
    let config = SeparatorConfig::default()
        .with_segment_length(0.5) // Process 0.5 second chunks
        .with_progress(false);

    println!("Loading model...");
    let separator = Arc::new(Separator::new(config)?);

    println!("Model loaded!");
    println!();
    println!("Note: This is a conceptual example.");
    println!("Real-time audio I/O requires platform-specific setup with CPAL.");
    println!();

    // Simulate processing a buffer
    let sample_rate: u32 = 44100;
    let channels = 2;
    let chunk_samples = sample_rate as usize / 2; // 0.5 seconds

    let audio_data = Array2::zeros((channels, chunk_samples));
    let audio = AudioBuffer::new(audio_data, sample_rate);

    println!("Processing simulated audio chunk...");
    let stems = separator.separate(&audio)?;

    println!("Separated {} stems:", stems.list().len());
    for name in stems.list() {
        if let Some(buffer) = stems.get(&name) {
            println!("  - {}: {} samples", name, buffer.samples());
        }
    }

    println!("\n✓ Simulation complete!");
    println!("\nFor real-time audio, integrate with CPAL:");
    println!("  1. Set up CPAL input stream");
    println!("  2. Buffer incoming audio");
    println!("  3. Process chunks with separator");
    println!("  4. Output to CPAL output stream");

    Ok(())
}
