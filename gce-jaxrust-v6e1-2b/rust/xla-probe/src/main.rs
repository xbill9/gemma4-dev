//! `xla-probe` — the Rust analogue of this rig's old `verify_jax_tpu`.
//!
//! Loading a PJRT plugin proves nothing. `dlopen` on `libtpu.so` succeeds on a host
//! with no chip attached, exactly as `import jax` succeeds with no TPU backend — which
//! is why the Python path asserted on `jax.devices()` rather than on the import. This
//! goes one step further than that and asserts on a *result*: it compiles a StableHLO
//! matmul, runs it on the device, and checks the number that comes back. A plugin that
//! loads, a client that creates and a device that lists are three separate things, and
//! none of them is evidence that the MXU computed anything.
//!
//! Exit codes: 0 = compiled and executed on a TPU device with the right answer;
//! 1 = anything else, with the stage that failed named on stderr.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use pjrt::{Client, HostBuffer, LoadedExecutable, Program, ProgramFormat};

/// Square matmul of two all-ones matrices: every output element is exactly K.
/// f32 because the check has to be exact — a bf16 accumulation of 256 ones is
/// still 256, but making the probe depend on that is the kind of assumption this
/// repo has been bitten by. The MXU is exercised either way.
const K: usize = 256;

const PROGRAM: &str = r#"
module @probe {
  func.func public @main(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %a, %b,
           contracting_dims = [1] x [0],
           precision = [DEFAULT, DEFAULT]
         : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    func.return %0 : tensor<256x256xf32>
  }
}
"#;

/// Where `libtpu.so` ends up, in the order worth trying.
///
/// `LIBTPU_PATH` first because that is the name `rlx-tpu` reads, and a rig that
/// disagrees with its own engine about which plugin to load is a bug waiting for a
/// capacity cycle to surface it. `TPU_LIBRARY_PATH` is what JAX reads, and both names
/// are in circulation on these VMs.
fn resolve_plugin(explicit: Option<String>) -> Result<PathBuf> {
    let mut tried: Vec<String> = Vec::new();

    if let Some(p) = explicit {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Ok(path);
        }
        bail!("plugin path `{p}` given on the command line does not exist");
    }

    for var in [
        "LIBTPU_PATH",
        "TPU_LIBRARY_PATH",
        "PJRT_PLUGIN_LIBRARY_PATH",
    ] {
        if let Ok(p) = std::env::var(var) {
            tried.push(format!("${var}={p}"));
            let path = PathBuf::from(&p);
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    // The libtpu wheel drops the plugin under site-packages. The startup script
    // installs it with a pinned interpreter, but the interpreter version moves, so
    // glob rather than hardcode one.
    for root in [
        "/usr/local/lib",
        "/usr/lib/python3/dist-packages",
        "/opt/libtpu",
    ] {
        for candidate in walk_for_libtpu(Path::new(root), 4) {
            tried.push(candidate.display().to_string());
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    Err(anyhow!(
        "no TPU PJRT plugin found. Set LIBTPU_PATH to libtpu.so, or pass it as argv[1]. Tried: {}",
        if tried.is_empty() {
            "nothing (no env var set, no site-packages copy)".to_string()
        } else {
            tried.join(", ")
        }
    ))
}

/// Shallow recursive search for `libtpu.so`; bounded depth so a probe never turns
/// into a filesystem walk on a VM with a large boot disk.
fn walk_for_libtpu(dir: &Path, depth: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    if depth == 0 {
        return found;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk_for_libtpu(&path, depth - 1));
        } else if path.file_name().is_some_and(|n| n == "libtpu.so") {
            found.push(path);
        }
    }
    found
}

fn main() -> Result<()> {
    let plugin_path = resolve_plugin(std::env::args().nth(1))?;
    println!("plugin      : {}", plugin_path.display());

    let api = pjrt::plugin(plugin_path.to_string_lossy().to_string())
        .load()
        .context("PJRT plugin loaded but GetPjrtApi failed")?;
    println!("api version : {:?}", api.version());

    let client = Client::builder(&api)
        .build()
        .context("PJRT_Client_Create failed — the plugin loaded but no runtime came up")?;
    println!(
        "platform    : {} {}",
        client.platform_name(),
        client.platform_version()
    );

    let devices = client.addressable_devices();
    if devices.is_empty() {
        bail!(
            "client created but exposes no addressable device — no chip is attached to this host"
        );
    }
    for device in &devices {
        let desc = device.description();
        let stats = device.memory_stats().ok();
        // largest_free_block_bytes is the number that decides whether a weight
        // tensor can actually be placed; bytes_limit alone has repeatedly looked
        // sufficient for a load that then failed on fragmentation.
        let hbm = match stats {
            Some(s) => format!(
                "limit {:.1} GiB, in use {:.1} GiB, largest free block {:.1} GiB",
                s.bytes_limit as f64 / (1 << 30) as f64,
                s.bytes_in_use as f64 / (1 << 30) as f64,
                s.largest_free_block_bytes as f64 / (1 << 30) as f64,
            ),
            None => "memory stats unavailable".to_string(),
        };
        println!(
            "device {:>2}   : {} — {}",
            i64::from(desc.id()),
            desc.kind(),
            hbm
        );
    }

    let platform = client.platform_name().to_string();
    if !platform.contains("tpu") {
        bail!("platform is `{platform}`, not TPU — this ran on the wrong backend");
    }

    let program = Program::new(ProgramFormat::MLIR, PROGRAM);
    let executable = LoadedExecutable::builder(&client, &program)
        .build()
        .context("StableHLO compile failed — the device is visible but XLA could not lower this")?;

    let ones = vec![1.0f32; K * K];
    let a = HostBuffer::builder()
        .data(ones.clone())
        .dims(vec![K as i64, K as i64])
        .build();
    let b = HostBuffer::builder()
        .data(ones)
        .dims(vec![K as i64, K as i64])
        .build();

    let inputs = vec![a.copy_to_sync(&client)?, b.copy_to_sync(&client)?];
    let result = executable
        .execution(inputs)
        .run_sync()
        .context("execution failed after a successful compile")?;

    let output = result
        .first()
        .and_then(|per_device| per_device.first())
        .ok_or_else(|| anyhow!("execution returned no output buffer"))?
        .copy_to_host_sync()?;

    let HostBuffer::F32(typed) = &output else {
        bail!("expected an f32 result, got {output:?}");
    };
    let got = typed.data();
    let expected = K as f32;
    let wrong = got.iter().filter(|v| (**v - expected).abs() > 1e-3).count();
    if wrong != 0 {
        bail!(
            "matmul returned the wrong numbers: {wrong}/{} elements differ from {expected} (first: {:?})",
            got.len(),
            got.first()
        );
    }

    println!(
        "matmul      : {K}x{K} @ {K}x{K} = {expected} everywhere ({} elements checked)",
        got.len()
    );
    println!("JAXRUST-PROBE: TPU compiled and executed StableHLO.");
    Ok(())
}
