//! Profile one separation run: stage timings and peak RSS.
//!
//! Usage: profile <input.wav> <model.onnx> [key=value ...]
//! Keys: preset=default|lowmem|maxspeed (applied first), threads=N,
//! opt=disable|basic|extended|all, mempattern=0|1, arena=0|1,
//! nofold=0|1 (disable ConstantFolding), shifts=N,
//! runs=N (repeat separation, default 1).
//! Prints one JSON line per run so results can be tabulated.

use charon_audio::{AudioFile, OnnxOptions, OptimizationLevel, Separator, SeparatorConfig};
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    // ru_maxrss is bytes on macOS and kilobytes on Linux.
    let bytes = if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64
    } else {
        usage.ru_maxrss as f64 * 1024.0
    };
    bytes / (1024.0 * 1024.0)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input.wav> <model.onnx> [key=value ...]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut config = SeparatorConfig::htdemucs(&args[2]).with_progress(false);
    let mut runs = 1;
    let mut label = Vec::new();
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        label.push(kv.clone());
        match k {
            "threads" => config.model.onnx.intra_threads = Some(v.parse()?),
            "opt" => {
                config.model.onnx.optimization_level = match v {
                    "disable" => OptimizationLevel::Disable,
                    "basic" => OptimizationLevel::Basic,
                    "extended" => OptimizationLevel::Extended,
                    _ => OptimizationLevel::All,
                }
            }
            "mempattern" => config.model.onnx.memory_pattern = v == "1",
            "arena" => config.model.onnx.cpu_arena = v == "1",
            "preset" => {
                config.model.onnx = match v {
                    "lowmem" => OnnxOptions::low_memory(),
                    "maxspeed" => OnnxOptions::max_speed(),
                    _ => OnnxOptions::default(),
                }
            }
            "nofold" => {
                config.model.onnx.disabled_optimizers = if v == "1" {
                    vec!["ConstantFolding".to_string()]
                } else {
                    Vec::new()
                }
            }
            "shifts" => config.process.shifts = v.parse()?,
            "runs" => runs = v.parse()?,
            other => anyhow::bail!("unknown key {other}"),
        }
    }

    eprintln!("onnx options: {:?}", config.model.onnx);
    let t = Instant::now();
    let audio = AudioFile::read(&args[1])?;
    let decode_s = t.elapsed().as_secs_f64();
    let rss_after_decode = peak_rss_mb();

    let t = Instant::now();
    let separator = Separator::new(config)?;
    let load_s = t.elapsed().as_secs_f64();
    let rss_after_load = peak_rss_mb();

    for run in 0..runs {
        let t = Instant::now();
        let stems = separator.separate(&audio)?;
        let sep_s = t.elapsed().as_secs_f64();
        let checksum: f64 = stems
            .list()
            .iter()
            .map(|n| {
                stems
                    .get(n)
                    .unwrap()
                    .data
                    .iter()
                    .map(|x| *x as f64)
                    .sum::<f64>()
            })
            .sum();
        println!(
            "{{\"label\":\"{}\",\"run\":{run},\"audio_s\":{:.2},\"decode_s\":{decode_s:.2},\"load_s\":{load_s:.2},\"separate_s\":{sep_s:.2},\"rtf\":{:.2},\"rss_decode_mb\":{rss_after_decode:.0},\"rss_load_mb\":{rss_after_load:.0},\"rss_peak_mb\":{:.0},\"checksum\":{checksum:.3}}}",
            label.join(" "),
            audio.duration(),
            audio.duration() / sep_s,
            peak_rss_mb(),
        );
    }
    Ok(())
}
