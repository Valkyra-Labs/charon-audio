//! `charon`: separate audio files, one-shot or through a resident server.
//!
//! ```text
//! charon separate <input> [-o DIR] [--model FILE] [--format F] [--shifts N] [--ep cpu|coreml|auto] [--overlap F] [--blend triangle|uniform] [--no-server]
//! charon serve --model FILE [--socket PATH] [--ep ...]
//! charon stop [--socket PATH]
//! ```
//!
//! `serve` loads the model once (for CoreML this includes the 7 s compiled
//! model load) and answers jobs on a Unix socket, one JSON object per line.
//! `separate` sends the job to a running server when the socket answers,
//! otherwise it runs in-process. All decoding, separation and writing happen
//! in the server process; the client only waits for the reply. The server
//! needs Unix sockets; elsewhere `separate` always runs in-process.

use charon_audio::{BitDepth, Blend, ExecutionProvider, Separator, SeparatorConfig, StemFormat};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct Job {
    input: PathBuf,
    output_dir: PathBuf,
    format: String,
    shifts: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind")]
enum Request {
    Separate(Job),
    Ping,
    Stop,
}

#[derive(Serialize, Deserialize)]
struct Reply {
    ok: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    stems: Vec<String>,
    #[serde(default)]
    seconds: f64,
    #[serde(default)]
    provider: String,
}

fn default_socket() -> PathBuf {
    let user = std::env::var("USER").unwrap_or_else(|_| "charon".to_string());
    std::env::temp_dir().join(format!("charon-{user}.sock"))
}

fn parse_format(name: &str) -> Result<StemFormat, String> {
    Ok(match name {
        "wav16" => StemFormat::Wav(BitDepth::Int16),
        "wav24" => StemFormat::Wav(BitDepth::Int24),
        "wav32" | "wav" => StemFormat::Wav(BitDepth::Float32),
        "flac16" => StemFormat::Flac(BitDepth::Int16),
        "flac24" | "flac" => StemFormat::Flac(BitDepth::Int24),
        other => return Err(format!("unknown format {other}")),
    })
}

fn parse_ep(name: &str) -> Result<ExecutionProvider, String> {
    Ok(match name {
        "cpu" => ExecutionProvider::Cpu,
        "coreml" => ExecutionProvider::CoreMl,
        "auto" => ExecutionProvider::Auto,
        other => return Err(format!("unknown execution provider {other}")),
    })
}

struct Options {
    positional: Vec<String>,
    model: Option<PathBuf>,
    output_dir: PathBuf,
    format: String,
    shifts: usize,
    ep: Option<ExecutionProvider>,
    overlap: Option<f32>,
    blend: Option<Blend>,
    threads: Option<usize>,
    socket: PathBuf,
    no_server: bool,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        positional: Vec::new(),
        model: std::env::var_os("CHARON_MODEL").map(PathBuf::from),
        output_dir: PathBuf::from("stems"),
        format: "wav32".to_string(),
        shifts: 1,
        ep: None,
        overlap: None,
        blend: None,
        threads: None,
        socket: std::env::var_os("CHARON_SOCKET")
            .map(PathBuf::from)
            .unwrap_or_else(default_socket),
        no_server: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match a.as_str() {
            "--model" | "-m" => o.model = Some(PathBuf::from(value("--model")?)),
            "--output" | "-o" => o.output_dir = PathBuf::from(value("--output")?),
            "--format" => o.format = value("--format")?,
            "--shifts" => {
                o.shifts = value("--shifts")?
                    .parse()
                    .map_err(|e| format!("--shifts: {e}"))?
            }
            "--ep" => o.ep = Some(parse_ep(&value("--ep")?)?),
            "--overlap" => {
                let v: f32 = value("--overlap")?
                    .parse()
                    .map_err(|e| format!("--overlap: {e}"))?;
                if !(0.0..1.0).contains(&v) {
                    return Err("--overlap must be in [0, 1)".to_string());
                }
                o.overlap = Some(v);
            }
            "--threads" => {
                o.threads = Some(
                    value("--threads")?
                        .parse()
                        .map_err(|e| format!("--threads: {e}"))?,
                )
            }
            "--blend" => {
                o.blend = Some(match value("--blend")?.as_str() {
                    "triangle" => Blend::Triangle,
                    "uniform" => Blend::Uniform,
                    other => return Err(format!("unknown blend {other}")),
                })
            }
            "--socket" => o.socket = PathBuf::from(value("--socket")?),
            "--no-server" => o.no_server = true,
            s if s.starts_with('-') => return Err(format!("unknown option {s}")),
            s => o.positional.push(s.to_string()),
        }
    }
    parse_format(&o.format)?;
    Ok(o)
}

/// Model contract from the file name: `*tiger*` is the TIGER music export,
/// `*split*` the split-transform HTDemucs export, anything else the
/// in-graph HTDemucs export.
fn config_for(o: &Options, model: &Path) -> SeparatorConfig {
    let ep = o.ep;
    let name = model.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut config = if name.contains("tiger") {
        SeparatorConfig::tiger_music(model)
    } else if name.contains("split") {
        SeparatorConfig::htdemucs_split(model)
    } else {
        SeparatorConfig::htdemucs(model)
    }
    .with_progress(false);
    if let Some(ep) = ep {
        config.model.onnx.execution_provider = ep;
    }
    if let Some(overlap) = o.overlap {
        config.process.overlap = overlap;
    }
    if let Some(blend) = o.blend {
        config.process.blend = blend;
    }
    if let Some(threads) = o.threads {
        config.model.onnx.intra_threads = Some(threads);
    }
    config
}

fn run_job(separator: &Separator, job: &Job) -> Result<Reply, String> {
    let t = Instant::now();
    let format = parse_format(&job.format)?;
    let stems = separator
        .separate_file(&job.input)
        .map_err(|e| e.to_string())?;
    stems
        .save_all_as(&job.output_dir, format)
        .map_err(|e| e.to_string())?;
    Ok(Reply {
        ok: true,
        message: String::new(),
        stems: stems.list(),
        seconds: t.elapsed().as_secs_f64(),
        provider: separator.provider().to_string(),
    })
}

#[cfg(not(unix))]
fn serve(_o: &Options) -> Result<(), String> {
    Err(NO_SERVER.to_string())
}

#[cfg(not(unix))]
fn send(_socket: &Path, _request: &Request) -> Result<Reply, String> {
    Err(NO_SERVER.to_string())
}

#[cfg(not(unix))]
const NO_SERVER: &str = "the resident server needs Unix sockets and is not available on this platform; `separate` runs in-process";

#[cfg(unix)]
fn serve(o: &Options) -> Result<(), String> {
    let model = o.model.clone().ok_or("serve needs --model")?;
    let t = Instant::now();
    let separator = Separator::new(config_for(o, &model)).map_err(|e| e.to_string())?;
    eprintln!(
        "charon serve: model loaded in {:.1} s on {}; socket {}",
        t.elapsed().as_secs_f64(),
        separator.provider(),
        o.socket.display()
    );
    let _ = std::fs::remove_file(&o.socket);
    let listener = UnixListener::bind(&o.socket).map_err(|e| e.to_string())?;
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let mut line = String::new();
        if BufReader::new(&stream).read_line(&mut line).is_err() {
            continue;
        }
        let reply = match serde_json::from_str::<Request>(&line) {
            Ok(Request::Ping) => Reply {
                ok: true,
                message: "pong".to_string(),
                stems: vec![],
                seconds: 0.0,
                provider: separator.provider().to_string(),
            },
            Ok(Request::Stop) => {
                let _ = writeln!(
                    stream,
                    "{}",
                    serde_json::json!({"ok": true, "message": "stopping"})
                );
                break;
            }
            Ok(Request::Separate(job)) => match run_job(&separator, &job) {
                Ok(r) => {
                    eprintln!(
                        "{} -> {} in {:.2} s",
                        job.input.display(),
                        job.output_dir.display(),
                        r.seconds
                    );
                    r
                }
                Err(message) => Reply {
                    ok: false,
                    message,
                    stems: vec![],
                    seconds: 0.0,
                    provider: String::new(),
                },
            },
            Err(e) => Reply {
                ok: false,
                message: format!("bad request: {e}"),
                stems: vec![],
                seconds: 0.0,
                provider: String::new(),
            },
        };
        let _ = writeln!(stream, "{}", serde_json::to_string(&reply).unwrap());
    }
    let _ = std::fs::remove_file(&o.socket);
    Ok(())
}

#[cfg(unix)]
fn send(socket: &Path, request: &Request) -> Result<Reply, String> {
    let mut stream = UnixStream::connect(socket).map_err(|e| e.to_string())?;
    writeln!(stream, "{}", serde_json::to_string(request).unwrap()).map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(&stream)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&line).map_err(|e| format!("bad reply: {e}"))
}

fn separate(o: &Options) -> Result<(), String> {
    let input = o.positional.first().ok_or("separate needs an input file")?;
    let job = Job {
        input: std::fs::canonicalize(input).map_err(|e| format!("{input}: {e}"))?,
        output_dir: std::path::absolute(&o.output_dir).map_err(|e| e.to_string())?,
        format: o.format.clone(),
        shifts: o.shifts,
    };
    let t = Instant::now();
    if !o.no_server {
        if let Ok(reply) = send(
            &o.socket,
            &Request::Separate(Job {
                input: job.input.clone(),
                output_dir: job.output_dir.clone(),
                format: job.format.clone(),
                shifts: job.shifts,
            }),
        ) {
            if !reply.ok {
                return Err(format!("server: {}", reply.message));
            }
            println!(
                "{} stems -> {} ({} on {}, {:.2} s server, {:.2} s total)",
                reply.stems.len(),
                job.output_dir.display(),
                reply.stems.join(", "),
                reply.provider,
                reply.seconds,
                t.elapsed().as_secs_f64()
            );
            return Ok(());
        }
    }
    let model = o.model.clone().ok_or("no server running; pass --model")?;
    let mut config = config_for(o, &model);
    config.process.shifts = o.shifts;
    let separator = Separator::new(config).map_err(|e| e.to_string())?;
    let reply = run_job(&separator, &job)?;
    println!(
        "{} stems -> {} ({} on {}, {:.2} s)",
        reply.stems.len(),
        job.output_dir.display(),
        reply.stems.join(", "),
        reply.provider,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    env_logger_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: charon separate <input> [-o DIR] [--model FILE] [--format wav16|wav24|wav32|flac16|flac24] [--shifts N] [--ep cpu|coreml|auto] [--overlap F] [--blend triangle|uniform] [--threads N] [--socket PATH] [--no-server]\n       charon serve --model FILE [--ep ...] [--socket PATH]\n       charon stop [--socket PATH]\n       charon ping [--socket PATH]";
    let result = match args.first().map(String::as_str) {
        Some("separate") => parse_args(&args[1..]).and_then(|o| separate(&o)),
        Some("serve") => parse_args(&args[1..]).and_then(|o| serve(&o)),
        Some("stop") => parse_args(&args[1..])
            .and_then(|o| send(&o.socket, &Request::Stop).map(|r| println!("{}", r.message))),
        Some("ping") => parse_args(&args[1..]).and_then(|o| {
            send(&o.socket, &Request::Ping).map(|r| println!("{} ({})", r.message, r.provider))
        }),
        _ => Err(usage.to_string()),
    };
    if let Err(e) = result {
        eprintln!("charon: {e}");
        std::process::exit(1);
    }
}

fn env_logger_init() {
    // Keep the binary free of env_logger; `log` output goes nowhere unless a
    // logger is installed. Errors are printed explicitly.
}
