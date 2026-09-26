//! Execution-provider probe: load any ONNX/ORT model, feed random inputs
//! shaped from the model's own metadata, run it N times and report time
//! per run, whether the output is finite, and the max difference from a
//! CPU run of the same input. For finding out which EP executes a graph
//! at all, and how fast, before any pipeline work.
//!
//! Usage: ep_probe <model> <ep: cpu|coreml|webgpu> [runs=N] [threads=N]
//!        [profile=<file prefix>] [nofold=1] [mempattern=0] [spec=fast]
//!        [units=all|gpu|ane] [format=mlprogram|nn] [rss=1] [log=verbose]
//!
//! `coreml` and `webgpu` need the `ep-experimental` feature.

#![cfg_attr(
    not(feature = "ep-experimental"),
    allow(unused_variables, unused_assignments, unused_mut)
)]

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, ValueType};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ep_probe <model> <cpu|coreml|webgpu> [runs=N] [threads=N] [profile=prefix] [nofold=1] [mempattern=0]");
        std::process::exit(1);
    }
    let (model, ep) = (&args[1], args[2].as_str());
    let mut runs = 3usize;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut profile = None;
    let mut nofold = false;
    let mut mempattern = true;
    let mut spec_fast = false;
    let mut units = "gpu".to_string();
    let mut format_nn = false;
    let mut rss = false;
    let mut verbose = false;
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "runs" => runs = v.parse()?,
            "threads" => threads = v.parse()?,
            "profile" => profile = Some(v.to_string()),
            "nofold" => nofold = v == "1",
            "mempattern" => mempattern = v == "1",
            "spec" => spec_fast = v == "fast",
            "units" => units = v.to_string(),
            "format" => format_nn = v == "nn",
            "rss" => rss = v == "1",
            "log" => verbose = v == "verbose",
            other => anyhow::bail!("unknown key {other}"),
        }
    }

    let build = |ep: &str| -> anyhow::Result<Session> {
        let mut b = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_memory_pattern(mempattern)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if nofold {
            b = b
                .with_disabled_optimizers("ConstantFolding")
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        if verbose {
            b = b
                .with_log_level(ort::logging::LogLevel::Verbose)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        if let Some(prefix) = &profile {
            b = b
                .with_profiling(format!("{prefix}_{ep}"))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let cpu = ort::ep::CPU::default().build();
        b = match ep {
            "cpu" => b.with_execution_providers([cpu]),
            #[cfg(feature = "ep-experimental")]
            "coreml" => {
                let cache = std::env::temp_dir().join("charon-coreml-cache");
                std::fs::create_dir_all(&cache)?;
                let mut ep = ort::ep::CoreML::default()
                    .with_model_format(if format_nn {
                        ort::ep::coreml::ModelFormat::NeuralNetwork
                    } else {
                        ort::ep::coreml::ModelFormat::MLProgram
                    })
                    .with_static_input_shapes(true)
                    .with_compute_units(match units.as_str() {
                        "all" => ort::ep::coreml::ComputeUnits::All,
                        "ane" => ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine,
                        _ => ort::ep::coreml::ComputeUnits::CPUAndGPU,
                    })
                    .with_model_cache_dir(cache.display());
                if spec_fast {
                    ep = ep.with_specialization_strategy(
                        ort::ep::coreml::SpecializationStrategy::FastPrediction,
                    );
                }
                b.with_execution_providers([ep.build(), cpu])
            }
            #[cfg(feature = "ep-experimental")]
            "webgpu" => b.with_execution_providers([ort::ep::WebGPU::default().build(), cpu]),
            other => anyhow::bail!("unknown or disabled EP {other}"),
        }
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let t = Instant::now();
        let s = b.commit_from_file(model)?;
        println!("{ep}: session built in {:.2} s", t.elapsed().as_secs_f64());
        Ok(s)
    };

    let mut session = build(ep)?;
    let mut inputs = Vec::new();
    let mut seed = 0x9E3779B97F4A7C15u64;
    for outlet in session.inputs() {
        let ValueType::Tensor { shape, .. } = outlet.dtype() else {
            anyhow::bail!("non-tensor input {}", outlet.name());
        };
        let dims: Vec<usize> = shape
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect();
        println!("input {} {:?}", outlet.name(), dims);
        inputs.push((outlet.name().to_string(), dims, data));
    }

    for outlet in session.outputs() {
        println!("output {} {:?}", outlet.name(), outlet.dtype());
    }

    let run = |session: &mut Session| -> anyhow::Result<(f64, Vec<f32>)> {
        let feed: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = inputs
            .iter()
            .map(|(name, dims, data)| {
                let t = Tensor::from_array((dims.clone(), data.clone())).unwrap();
                (
                    std::borrow::Cow::from(name.as_str()),
                    ort::session::SessionInputValue::from(t),
                )
            })
            .collect();
        let t = Instant::now();
        let out = session.run(feed)?;
        let secs = t.elapsed().as_secs_f64();
        let (_, values) = out[0].try_extract_tensor::<f32>()?;
        Ok((secs, values.to_vec()))
    };

    let mut last = None;
    for i in 0..runs {
        let (secs, values) = run(&mut session)?;
        let finite = values.iter().all(|x| x.is_finite());
        println!(
            "{ep} run {i}: {secs:.3} s, output {} values, finite {finite}",
            values.len()
        );
        last = Some(values);
    }
    if profile.is_some() {
        let path = session.end_profiling()?;
        println!("profile written: {path}");
    }
    #[cfg(unix)]
    if rss {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        println!(
            "peak RSS {:.0} MB",
            usage.ru_maxrss as f64 / (1024.0 * 1024.0)
        );
    }
    #[cfg(not(unix))]
    if rss {
        println!("peak RSS: not measured on this platform");
    }
    if ep != "cpu" {
        let mut cpu = build("cpu")?;
        let (secs, reference) = run(&mut cpu)?;
        let out = last.unwrap();
        let max = out
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("cpu reference: {secs:.3} s; max |{ep} - cpu| = {max:.3e}");
    }
    Ok(())
}
