"""Offline regression tests for the G5g llama.cpp MCP server.

No AWS, no network, no GPU, no compiler.

**THE POINT OF THIS FILE IS THE RENDERED BOOTSTRAP.** When `gpu-pytorch-g5g-2b`
was forked from the JAX rig it shipped FIVE fatal bugs that all survived to the
first real launch — `import jax` in a rig with no jax, an ExecStart naming a file
not in the payload, three serve flags the server did not define, a quoted pip
spec holding two packages, and `torch.compile(backend="tpu")` on CUDA. 89 offline
tests passed throughout, because **not one of them asserted on the rendered
bootstrap**. A fork rewrites the parts you read and leaves the parts you execute.

So the bootstrap tests below are not decoration. They are the tests.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

# Spelled out rather than read from `server`, so a change to the served artifact
# shows up as a test edit rather than passing vacuously against whatever the
# module happens to hold.
REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
GGUF = "gemma-4-E2B_q4_0-it.gguf"
DEFAULT_SIZE = "g5g.2xlarge"
SWAP_SIZE = "g5g.xlarge"

# Flags `llama-server` actually defines. _serve_argv must emit only these.
KNOWN_FLAGS = {
    "--hf-repo", "--hf-file", "--host", "--port", "--ctx-size",
    "--parallel", "--n-gpu-layers", "--metrics", "--mmproj-file",
}

# Flags that exist ONLY in this repo's own JAX/PyTorch servers. The PyTorch fork
# emitted three of these against a server that defined none of them.
FOREIGN_FLAGS = {
    "--quant-mode", "--ple-bits", "--int8-lm-head", "--prefill-chunk-size",
    "--kv-cache-dtype", "--max-model-len", "--seq", "--model",
}


class ServeArgv(unittest.TestCase):
    def setUp(self):
        self.argv = server._serve_argv(REPO, DEFAULT_SIZE)

    def test_only_emits_flags_llama_server_defines(self):
        emitted = set(re.findall(r"--[a-z0-9-]+", self.argv))
        self.assertTrue(
            emitted <= KNOWN_FLAGS,
            f"argv emits flags llama-server does not define: {emitted - KNOWN_FLAGS}",
        )

    def test_emits_no_flag_from_a_sibling_rig(self):
        emitted = set(re.findall(r"--[a-z0-9-]+", self.argv))
        self.assertEqual(
            emitted & FOREIGN_FLAGS, set(),
            "argv carries a flag from the JAX/PyTorch siblings; argparse exits 2 "
            "on an unknown flag and the unit crash-loops from the first start.",
        )

    def test_names_both_halves_of_the_artifact(self):
        # Slot 5 of the directory name is a claim about WHICH FILE, and a GGUF
        # repo can hold several quantisations. Naming the repo alone would let
        # llama-server pick.
        self.assertIn(f"--hf-repo {REPO}", self.argv)
        self.assertIn(f"--hf-file {GGUF}", self.argv)

    def test_metrics_is_always_on(self):
        # Off by default in llama.cpp. Without it /metrics 404s and the decode
        # gauge this whole rig family compares on does not exist.
        self.assertIn("--metrics", self.argv)

    def test_offloads_every_layer(self):
        self.assertIn(f"--n-gpu-layers {server.N_GPU_LAYERS}", self.argv)

    def test_projector_is_absent_unless_configured(self):
        self.assertNotIn("--mmproj-file", self.argv)


class RenderedBootstrap(unittest.TestCase):
    """The tests the PyTorch fork did not have."""

    @classmethod
    def setUpClass(cls):
        cls.ud = server._user_data(REPO, DEFAULT_SIZE)
        cls.code = [
            line for line in cls.ud.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def test_it_renders_at_all(self):
        # str.format() raises on a stray literal brace, which is how a bootstrap
        # edit breaks the deploy without touching anything that looks like code.
        self.assertGreater(len(self.ud), 2000)

    def test_no_executable_line_mentions_torch_jax_or_pip(self):
        # The literal fork bug: `import jax` in verify_gpu on a rig that installs
        # no jax. Under `set -e` it killed install.sh, INSTALL_DONE was never
        # written, and get_install_progress reported "IN PROGRESS" forever.
        bad = [
            line for line in self.code
            if re.search(r"\btorch\b|\bjax\b|PYTHON_BIN|pip install", line, re.I)
        ]
        self.assertEqual(bad, [], f"stale runtime in the bootstrap: {bad}")

    def test_execstart_points_at_the_binary_the_build_installs(self):
        # The second fork bug: ExecStart named a file that was not in the
        # payload, so the unit crash-looped.
        self.assertIn(f"ExecStart={server.APP_DIR}/llama-server", self.ud)
        self.assertIn(f"install -m 0755 {server.BUILD_DIR}/build/bin/llama-server", self.ud)

    def test_cuda_is_required_not_merely_requested(self):
        # cmake's default on a missing toolkit is to carry on WITHOUT CUDA and
        # produce a working CPU-only llama-server. -DGGML_CUDA=ON makes that a
        # configure failure instead.
        self.assertIn("-DGGML_CUDA=ON", self.ud)

    def test_cuda_arch_is_pinned_to_the_only_target(self):
        self.assertIn(f"-DCMAKE_CUDA_ARCHITECTURES={server.CUDA_ARCH}", self.ud)
        self.assertEqual(server.CUDA_ARCH, "75", "T4G is SM 7.5")

    def test_curl_is_on_or_hf_download_does_not_exist(self):
        # --hf-repo/--hf-file are gated behind LLAMA_CURL. Without it the build
        # succeeds and the serve fails on an unrecognised argument.
        self.assertIn("-DLLAMA_CURL=ON", self.ud)

    def test_nvcc_probe_searches_the_dlami_pip_tree(self):
        """MEASURED 2026-09-02 on i-05005979aff5a0df9 -- the rig's first launch.

        The ARM64 PyTorch DLAMI has **no /usr/local/cuda and no nvcc on PATH**, so
        the original two-location probe exited 1 and killed the install. The
        toolkit is there, as a pip package inside the PyTorch venv at
        /opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc.

        Globbed, never hardcoded: the path moves with the DLAMI's Python version.
        """
        self.assertIn("site-packages/nvidia/cu*/bin/nvcc", self.ud)
        self.assertIn("/usr/local/cuda-*/bin/nvcc", self.ud)

    def test_versioned_sonames_get_dev_symlinks(self):
        """pip wheels ship libcublas.so.13 but not plain libcublas.so, and
        cmake's FindCUDAToolkit wants the bare name. Without these the configure
        fails to find cuBLAS with the library sitting right there."""
        self.assertIn("ln -sf", self.ud)
        self.assertIn("cublas", self.ud)

    def test_cmake_is_told_the_toolkit_explicitly(self):
        """Environment alone does not steer FindCUDAToolkit reliably."""
        self.assertIn("-DCMAKE_CUDA_COMPILER=", self.ud)
        self.assertIn("-DCUDAToolkit_ROOT=", self.ud)

    def test_runtime_library_path_is_rewritten_after_the_probe(self):
        """The CUDA libs are not on the default loader path, so llama-server
        cannot start without this. Rewritten post-probe because the root is only
        known then -- the same discipline as the PyTorch sibling's ExecStart."""
        self.assertIn("LD_LIBRARY_PATH=", self.ud)
        self.assertIn("sed -i \"s|^LD_LIBRARY_PATH=", self.ud)

    def test_nvcc_absence_is_fatal(self):
        # A CPU-only build SERVES CORRECTLY and is several times slower, so the
        # honest outcome is a dead install rather than a misleading rig.
        self.assertIn("FATAL: no nvcc", self.ud)
        self.assertRegex(self.ud, r"FATAL: no nvcc[\s\S]{0,600}?exit 1")

    def test_the_built_binary_is_asserted_to_see_cuda(self):
        # nvidia-smi succeeding proves the driver works, not that what we
        # compiled can use it. Assert on the binary's own device list.
        self.assertIn("--list-devices", self.ud)
        self.assertIn("lists no CUDA device", self.ud)

    def test_verify_runs_before_the_unit_is_enabled(self):
        # Ordering is the whole point: a CPU-only build starts and binds fine.
        self.assertLess(
            self.ud.index("verify_gpu\n"),
            self.ud.index(f"systemctl enable --now {server.SERVICE_NAME}"),
        )

    def test_install_done_is_written_only_after_verification(self):
        # Match the actual `touch`, not the substring: an apt comment upstream
        # says "INSTALL_DONE is never touched" and would satisfy a loose search.
        self.assertLess(
            self.ud.index("verify_gpu\n"),
            self.ud.index(f"touch {server.APP_DIR}/INSTALL_DONE"),
        )

    def test_unattended_upgrades_is_masked(self):
        # MEASURED on the sibling: it restarts services it upgrades, mid-install,
        # and reports Result=success NRestarts=0 while doing it. Cost two full
        # checkpoint downloads and looked exactly like an OOM.
        self.assertIn("mask apt-daily-upgrade.service", self.ud)

    def test_start_timeout_survives_the_first_download(self):
        # First start pulls 3.35 GB before it binds. systemd's default would kill
        # it mid-download and restart it, re-downloading from zero.
        self.assertIn("TimeoutStartSec=", self.ud)

    def test_hf_token_fetch_is_not_traced(self):
        # The script runs under `set -x` and bash traces assignments WITH their
        # values, so the token would land in a world-readable cloud-init log.
        i = self.ud.index("secretsmanager get-secret-value")
        before, after = self.ud[:i], self.ud[i:]
        self.assertIn("set +x", before)
        self.assertIn("set -x", after)

    def test_token_is_never_in_user_data_itself(self):
        self.assertNotIn("HF_TOKEN=hf_", self.ud)

    def test_swapfile_renders_for_every_size_at_or_below_16gib(self):
        # The DEFAULT size gets a swapfile. Worth pinning, because the sibling
        # made this threshold inclusive and thereby rendered the block for
        # g5g.2xlarge for the first time -- which is how its `mkswap -q` bug
        # finally fired, months after being written.
        self.assertIn("mkswap", server._user_data(REPO, SWAP_SIZE))
        self.assertIn("mkswap", self.ud)
        self.assertNotIn("mkswap", server._user_data(REPO, "g5g.4xlarge"))

    def test_mkswap_takes_no_busybox_flag(self):
        # `mkswap -q` is busybox; util-linux rejects it, and under `set -e` that
        # killed cloud-init before install.sh was even written.
        #
        # Matched against EXECUTABLE lines only: the bootstrap comment explaining
        # this bug contains the literal string, so a naive assertNotIn over the
        # whole script fails on its own documentation.
        code = [
            line for line in server._user_data(REPO, SWAP_SIZE).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertEqual([line for line in code if "mkswap -q" in line], [])

    def test_stage_markers_exist_for_every_long_phase(self):
        # On spot the build is what gets reclaimed; without markers nothing
        # records which step had been reached.
        for stage in ("build-deps", "nvcc-probe", "cmake-configure",
                      "cmake-build", "install-binary", "gpu-verify"):
            self.assertIn(f"stage {stage}", self.ud)


class MetricsShim(unittest.TestCase):
    EXPO = (
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
        'llamacpp:prompt_tokens_total{model="gemma"} 512\n'
        'llamacpp:prompt_seconds_total{model="gemma"} 2.0\n'
        'llamacpp:tokens_predicted_total{model="gemma"} 300\n'
        'llamacpp:tokens_predicted_seconds_total{model="gemma"} 10.0\n'
        'llamacpp:n_busy_slots_per_decode{model="gemma"} 1.0\n'
    )

    def test_parses_and_hoists_the_model_label(self):
        samples, model = server._parse_prom(self.EXPO)
        self.assertEqual(model, "gemma")
        self.assertEqual(samples["llamacpp:tokens_predicted_total"], 300.0)

    def test_decode_rate_excludes_prefill(self):
        samples, _ = server._parse_prom(self.EXPO)
        # 300 generated tokens / 10s of DECODE = 30 tok/s. Folding in the 512
        # prompt tokens and 2s of prefill would give 67.7 -- a different and
        # incomparable number.
        self.assertAlmostEqual(server._decode_rate(samples), 30.0)

    def test_decode_rate_is_none_when_nothing_has_been_generated(self):
        samples, _ = server._parse_prom("llamacpp:tokens_predicted_total 0\n")
        self.assertIsNone(server._decode_rate(samples))

    def test_comments_and_junk_are_ignored(self):
        samples, _ = server._parse_prom("# HELP x\n\nnot_a_metric abc\n")
        self.assertEqual(samples, {})


class Identity(unittest.TestCase):
    def test_rig_name_matches_the_directory(self):
        # The directory name is the key everything derives from: MCP server name,
        # skill stem, ManagedBy tag, zone cache. NAMING.md.
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_rig_name_parses_as_five_slots(self):
        slots = server.RIG_NAME.split("-")
        self.assertEqual(len(slots), 5, "platform-runtime-hardware-model-encoding")
        self.assertEqual(slots[0], "gpu")
        self.assertEqual(slots[1], "llamacpp")   # never `llama`, never `llama-cpp`
        self.assertEqual(slots[4], "q4_0")       # the encoding, not `gguf`

    def test_managed_by_tag_scopes_discovery_to_this_rig(self):
        self.assertEqual(server.MANAGED_BY, server.RIG_NAME)

    def test_build_id_names_both_ref_and_arch(self):
        # The same ref built for a different arch is a different binary.
        self.assertIn(server.LLAMA_CPP_REF, server._build_id())
        self.assertIn(server.CUDA_ARCH, server._build_id())

    def test_llama_cpp_ref_is_pinned_not_a_branch(self):
        self.assertNotIn(server.LLAMA_CPP_REF, ("master", "main", "HEAD"))
        self.assertRegex(server.LLAMA_CPP_REF, r"^b\d+$")

    def test_no_payload_machinery_survived_the_fork(self):
        for gone in ("_PAYLOAD_FILES", "_payload_root", "_payload_digest",
                     "_payload_tar_b64", "deploy_torch_server"):
            self.assertFalse(
                hasattr(server, gone),
                f"{gone} should not exist: this rig has no serving payload.",
            )


class EnvIsHonest(unittest.TestCase):
    """tpu.env must not carry keys nothing reads.

    The PyTorch fork inherited QUANT_MODE/PLE_BITS/INT8_LM_HEAD/DTYPE and carried
    them for a month, until its own CLAUDE.md had to warn that they were inert.
    A key present in an env file reads as a supported knob.
    """

    def test_every_key_is_read_by_server(self):
        env = [
            line.split("=", 1)[0]
            for line in (ROOT / "tpu.env").read_text().splitlines()
            if re.match(r"^[A-Z_]+=", line)
        ]
        read = set(re.findall(r'os\.getenv\(\s*"([A-Z_]+)"', (ROOT / "server.py").read_text()))
        inert = [k for k in env if k not in read]
        self.assertEqual(inert, [], f"tpu.env keys nothing reads: {inert}")

    def test_env_names_the_gguf_file_not_only_the_repo(self):
        text = (ROOT / "tpu.env").read_text()
        self.assertIn(f"MODEL_FILE={GGUF}", text)


class SkillSnapshot(unittest.TestCase):
    def test_snapshot_list_carries_no_serving_payload(self):
        # Read the `names` tuple the script actually copies. refresh_skill.py's
        # comment names the sibling payload files it deliberately dropped, so
        # searching the whole file matches its own explanation.
        import ast as _ast
        tree = _ast.parse((ROOT / "refresh_skill.py").read_text())
        names = [
            _ast.literal_eval(node.value)
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Assign)
            and getattr(node.targets[0], "id", None) == "names"
        ]
        self.assertEqual(len(names), 1, "expected exactly one `names` tuple")
        self.assertEqual(
            set(names[0]), {"server.py", "project-setup.sh", "requirements.txt"}
        )


if __name__ == "__main__":
    unittest.main()
