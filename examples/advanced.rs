//! Example: Advanced audio processing with performance optimizations

use charon_audio::{
    AudioKNN, BatchProcessor, PerformanceHint, PerformanceHints, SimdOps,
};
use ndarray::Array2;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("Charon Advanced Audio Processing");
    println!("=================================\n");

    // Demonstrate performance hints
    demo_performance_hints();

    // Demonstrate SIMD operations
    demo_simd_operations();

    // Demonstrate KNN audio similarity
    demo_audio_similarity()?;

    // Demonstrate batch processing
    demo_batch_processing()?;

    Ok(())
}

fn demo_performance_hints() {
    println!("1. Performance Hints");
    println!("   ----------------");

    let hints = PerformanceHints::new()
        .with_hint(PerformanceHint::Parallel)
        .with_hint(PerformanceHint::Vectorize)
        .with_hint(PerformanceHint::CacheFriendly)
        .with_hint(PerformanceHint::LowLatency);

    println!("   Enabled hints:");
    if hints.has_hint(PerformanceHint::Parallel) {
        println!("     ✓ Parallel processing");
    }
    if hints.has_hint(PerformanceHint::Vectorize) {
        println!("     ✓ SIMD vectorization");
    }
    if hints.has_hint(PerformanceHint::CacheFriendly) {
        println!("     ✓ Cache-friendly access");
    }
    if hints.has_hint(PerformanceHint::LowLatency) {
        println!("     ✓ Low-latency optimization");
    }
    println!();
}

fn demo_simd_operations() {
    println!("2. SIMD-Optimized Operations");
    println!("   --------------------------");

    // Generate test audio samples
    let mut samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
    let window: Vec<f32> = (0..1000)
        .map(|i| 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / 999.0).cos()))
        .collect();

    let start = Instant::now();

    // SIMD multiply (apply window)
    SimdOps::multiply(&mut samples, &window);

    // SIMD RMS calculation
    let rms = SimdOps::rms(&samples);

    // Peak detection
    let peaks = SimdOps::find_peaks(&samples, 0.3);

    let elapsed = start.elapsed();

    println!("   Processed 1000 samples in {elapsed:?}");
    println!("   RMS: {rms:.4}");
    println!("   Peaks detected: {}", peaks.len());
    println!();
}

fn demo_audio_similarity() -> anyhow::Result<()> {
    println!("3. KNN Audio Similarity Search");
    println!("   ----------------------------");

    // Create synthetic audio data (2 channels, 5000 samples)
    let audio = Array2::from_shape_fn((2, 5000), |(ch, i)| {
        (i as f32 * 0.01 + ch as f32).sin() * 0.5
    });

    // Define a query segment (first 500 samples)
    let query_segment = audio.slice(ndarray::s![.., 0..500]);

    let knn = AudioKNN::new(5);
    let start = Instant::now();

    // Find similar segments
    let similar = knn.find_similar_segments(
        query_segment,
        audio.view(),
        100, // hop size
    );

    let elapsed = start.elapsed();

    println!("   Found {} similar segments in {:?}", similar.len(), elapsed);
    for (i, (position, distance)) in similar.iter().take(3).enumerate() {
        println!(
            "     {}. Position: {} samples, Distance: {:.4}",
            i + 1,
            position,
            distance
        );
    }
    println!();

    Ok(())
}

fn demo_batch_processing() -> anyhow::Result<()> {
    println!("4. Parallel Batch Processing");
    println!("   --------------------------");

    // Simulate multiple audio files
    let files: Vec<String> = (0..20)
        .map(|i| format!("audio_{i}.wav"))
        .collect();

    let processor = BatchProcessor::new(5).with_threads(4);

    let start = Instant::now();

    // Process in parallel (simulated)
    let results = processor.process(files, |filename| {
        // Simulate processing
        std::thread::sleep(std::time::Duration::from_millis(10));
        format!("Processed: {filename}")
    });

    let elapsed = start.elapsed();

    println!("   Processed {} files in {:?}", results.len(), elapsed);
    println!(
        "   Average: {:?} per file",
        elapsed / results.len() as u32
    );
    println!();

    Ok(())
}

#[allow(dead_code)]
fn demo_full_pipeline() -> anyhow::Result<()> {
    println!("5. Full Separation Pipeline");
    println!("   -------------------------");

    // This would require an actual model file
    // Commented out as it's not runnable without a model

    /*
    let config = SeparatorConfig::onnx("model.onnx")
        .with_shifts(2)
        .with_segment_length(5.0)
        .with_progress(true);

    let separator = Separator::new(config)?;

    let start = Instant::now();
    let stems = separator.separate_file("test_audio.mp3")?;
    let elapsed = start.elapsed();

    println!("   Separation completed in {:?}", elapsed);
    println!("   Stems extracted: {:?}", stems.list());
    */

    Ok(())
}
