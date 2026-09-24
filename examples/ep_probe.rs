//! Execution-provider probe: load any ONNX/ORT model, feed random inputs
//! shaped from the model's own metadata, run it N times and report time
//! per run, whether the output is finite, and the max difference from a
//! CPU run of the same input. For finding out which EP executes a graph
//! at all, and how fast, before any pipeline work.
//!
//! Usage: ep_probe <model> <ep: cpu|coreml|webgpu> [runs=N] [threads=N]
//!        [profile=<file prefix>] [nofold=1] [mempattern=0]
//!
//! `coreml` and `webgpu` need the `ep-experimental` feature.

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
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "runs" => runs = v.parse()?,
            "threads" => threads = v.parse()?,
            "profile" => profile = Some(v.to_string()),
            "nofold" => nofold = v == "1",
            "mempattern" => mempattern = v == "1",
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
                b.with_execution_providers([
                    ort::ep::CoreML::default()
                        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                        .with_static_input_shapes(true)
                        .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                        .with_model_cache_dir(cache.display())
                        .build(),
                    cpu,
                ])
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
