//! `tpu-selftest` — does rlx actually execute on this accelerator?
//!
//! This exists because `xla-probe` turned out **not** to answer that question, and the
//! difference cost a deployment to find. The probe talks to the PJRT C API through the
//! `pjrt` crate; the engine talks to it through `rlx-tpu`'s own hand-written bindings.
//! Those are two different ABIs of two different vintages, and on 2026-08-28 they
//! disagreed: `pjrt-sys` 0.2.0 builds `PJRT_Client_Create_Args` with 9 fields (72 bytes)
//! while libtpu 0.75 requires the 11-field, 88-byte layout that added
//! `kKeyValueTryGetCallback` in PJRT API 0.59. The probe failed at
//! `PJRT_Client_Create` — and that failure said nothing whatsoever about the engine,
//! because `rlx-tpu` already carries the 88-byte struct.
//!
//! So this binary lives in the engine crate and links the engine's own stack. It is a
//! strict subset of what the server does: same rlx, same backend, same libtpu — minus
//! the model, the weights and the HTTP layer. That is what "a smaller question" has to
//! mean to be worth anything.
//!
//! Exit 0 = rlx compiled a graph for the device and the arithmetic came back right.
//!
//! ## The teardown segfault (MEASURED on a v6e-1, 2026-08-28)
//!
//! rlx-tpu 0.2.11 against libtpu 0.75 computes correctly and then **SIGSEGVs while
//! dropping the compiled graph**. Every line of output — including the success marker —
//! is printed first, stderr is empty, and there is no Rust panic; the process simply
//! dies with 139 after `main` returns. Leaving via `std::process::exit`, which skips
//! destructors, exits 0 cleanly on the identical binary and inputs. That is the whole
//! diagnosis: the failure is in PJRT client teardown, not in compilation or execution.
//!
//! This binary therefore exits without unwinding by default, and `SELFTEST_RUN_DROP=1`
//! reproduces the crash for anyone re-testing it against a newer rlx or libtpu.
//!
//! **The reason this matters beyond a tidy exit code:** a marker-scanning supervisor
//! sees a process that printed every success line it was supposed to. Only the exit
//! status distinguishes it from a healthy run — which is why `verify_rust_tpu` asserts
//! on the return code and never on the presence of the marker.

use anyhow::{bail, Result};
use rlx::ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// Square matmul of two all-ones matrices: every output element must equal K exactly.
/// Small enough to compile fast, large enough to go through the MXU rather than a
/// scalar fallback.
const K: usize = 256;

fn main() -> Result<()> {
    let device = match std::env::args().nth(1).as_deref() {
        Some("cpu") => Device::Cpu,
        Some("tpu") | None => Device::Tpu,
        Some(other) => bail!("unknown device `{other}` — use `tpu` or `cpu`"),
    };
    println!("device      : {device:?}");

    let mut g = Graph::new("selftest");
    let a = g.input("a", Shape::new(&[K, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, K], DType::F32));
    let y = g.matmul(a, b, Shape::new(&[K, K], DType::F32));
    g.set_outputs(vec![y]);
    println!("graph       : {K}x{K} @ {K}x{K} matmul, f32");

    // Compiling is the step that reaches the device: it lowers the IR to HLO and hands
    // it to libtpu. A failure here is a backend problem, not an arithmetic one.
    let mut compiled = Session::new(device).compile(g);
    println!("compiled    : ok");

    let ones = vec![1.0f32; K * K];
    compiled.set_param("b", &ones);
    let out = compiled.run(&[("a", &ones)]);

    let Some(result) = out.first() else {
        bail!("run returned no outputs");
    };
    if result.len() != K * K {
        bail!("expected {} elements, got {}", K * K, result.len());
    }

    // Assert on the numbers, not on the absence of an error. A backend that silently
    // falls back, or one that returns a zero-filled buffer, passes every check except
    // this one.
    let expected = K as f32;
    let wrong = result.iter().filter(|v| (**v - expected).abs() > 1e-3).count();
    if wrong != 0 {
        bail!(
            "matmul returned the wrong numbers: {wrong}/{} elements differ from {expected} (first: {:?})",
            result.len(),
            result.first()
        );
    }

    println!("matmul      : {expected} everywhere ({} elements checked)", result.len());
    println!("JAXRUST-SELFTEST: rlx compiled and executed on {device:?}.");

    if std::env::var("SELFTEST_RUN_DROP").is_ok() {
        // Opt in to the crash, to re-test whether it still happens.
        return Ok(());
    }
    // Flush explicitly: process::exit does not run atexit handlers or flush Rust's
    // stdout buffer, and losing the success line to a skipped flush would be a
    // self-inflicted version of exactly the bug above.
    use std::io::Write;
    std::io::stdout().flush().ok();
    std::process::exit(0);
}
