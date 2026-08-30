"""Offline regression tests for the local-CPU JAX MCP server.

No network, no accelerator, no cloud account. These pin the facts that make this
rig different from its siblings, and almost all of those differences are
subtractions: there is no control plane, no provisioning, no instance, no
credentials, and no deploy step. A test suite forked from a cloud rig will pass
happily while asserting things about a machine that does not exist, so the first
class below asserts the ABSENCE of that vocabulary rather than trusting that the
rewrite was complete.

What is left is the part that was never about the cloud: the dtype policy, the
memory arithmetic, the degeneracy guard, the payload digest, the observability
machinery, and the load-time quantizers. Those carry, and they are the reason
this rig is worth having — it is the only place they can be exercised end to end
without spending a capacity cycle.

The engine tests import ports/gemma4 under JAX_PLATFORMS=cpu, which here is not
a simulation of anything: it is the rig.
"""

import asyncio
import filecmp
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

# Spelled out rather than read from server, so that a change to the served
# checkpoint shows up as a test edit rather than passing vacuously against
# whatever the module happens to hold.
MODEL = "google/gemma-4-E2B-it"

# The eight files refresh_skill.py snapshots into both skill copies: the MCP
# control plane *and* the serving payload, because an installed skill still has
# to be able to start a serve.
SKILL_SOURCES = (
    "server.py", "project-setup.sh", "requirements.txt",
    "requirements-serving.txt", "jax_openai_server.py", "jax_engine.py",
    "ports/gemma4/jax_e_loader.py", "ports/gemma4/jax_e_model.py",
)

GB = 1_000_000_000


def run(coro):
    return asyncio.run(coro)


def tpu_env():
    values = {}
    for line in (ROOT / "tpu.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def code_only(path):
    """Source with comments and docstrings BLANKED OUT, layout preserved.

    Every "this word must not appear" test below needs this, and the reason is
    not pedantry: the most valuable prose in this rig is precisely the prose
    that NAMES the thing that was removed. server.py's docstring explains that
    there is no spot reclamation to handle; the daemon launch carries a long
    comment about why it is NOT asyncio.create_subprocess_exec; jax_engine's RSS
    helper says "deliberately not psutil". A naive substring search fails on all
    three, and the obvious fix — deleting the explanation — makes the codebase
    worse. So strip the prose and search the code.

    It blanks characters in place rather than re-joining tokens. The first
    version joined token strings with newlines, which silently broke every
    multi-token search: `shell=True` became "shell", "=", "True" on three lines,
    so the test asserting its absence could never have failed. A test that
    cannot fail is worse than no test, and this one was guarding a rule the
    monorepo states explicitly.
    """
    import io
    import tokenize

    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    blanks = []
    prev = tokenize.NEWLINE
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type == tokenize.COMMENT:
                blanks.append((tok.start, tok.end))
            elif tok.type == tokenize.STRING and prev in (
                tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
            ):
                blanks.append((tok.start, tok.end))  # a docstring, not a value
            if tok.type not in (tokenize.NL, tokenize.COMMENT):
                prev = tok.type
    for (srow, scol), (erow, ecol) in reversed(blanks):
        for row in range(srow, erow + 1):
            line = lines[row - 1]
            a = scol if row == srow else 0
            b = ecol if row == erow else len(line.rstrip("\n"))
            lines[row - 1] = line[:a] + " " * (b - a) + line[b:]
    return "".join(lines)


class Facts(dict):
    """A _host_facts() reading, for driving the capacity arithmetic offline."""

    def __init__(self, ram_available, swap_free, ram_total=15 * GB,
                 swap_total=None, cores=16, disk_free=200 * GB):
        super().__init__(
            cores=cores, ram_total=ram_total, ram_available=ram_available,
            swap_total=swap_total if swap_total is not None else swap_free,
            swap_free=swap_free, disk_free=disk_free,
        )


class HostFactsMixin:
    """Swap in a synthetic host so the verdicts are tested, not this machine's."""

    def withFacts(self, facts):
        saved = server._host_facts
        server._host_facts = lambda: facts
        self.addCleanup(lambda: setattr(server, "_host_facts", saved))


class BashSyntaxMixin:
    def assertScriptParses(self, path):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"{path.name}: {proc.stderr}")


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "check_host_capacity", "check_dependencies", "verify_cpu_backend",
            "fetch_checkpoint", "get_serve_command", "start_jax_server",
            "stop_jax_server", "list_jax_servers", "get_jax_logs", "get_endpoint",
            "verify_model_health", "query_model", "get_metrics", "save_hf_token",
            "get_help",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        # Only stopping. There is nothing here whose loss is expensive: nothing
        # was built, and the warm compilations survive in a disk cache that is
        # NOT ephemeral on this rig.
        self.assertEqual(destructive, {"stop_jax_server"})
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool.title)
                self.assertTrue(tool.description)
                self.assertIsNotNone(tool.annotations)

    def test_no_tool_takes_an_instance_id(self):
        # Every cloud sibling threads instance_id through every tool. Carrying
        # one here would be the clearest sign the fork was incomplete.
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertNotIn("instance_id", tool.inputSchema.get("properties", {}))


class NoCloudControlPlaneTests(unittest.TestCase):
    """The rewrite is a subtraction, so assert the subtraction actually happened.

    A fork of a cloud rig keeps passing its own tests while describing hardware
    that does not exist — that is precisely how `gpu-jax-g4dn-2b` shipped with
    four registration files naming `gpu-jax-g5g-2b` and a skill path that was
    never there. Dead cloud code is worse here than merely unused: it reads as
    live configuration.
    """

    FORBIDDEN = ("boto3", "botocore", "ssm", "SecretsManager", "secretsmanager",
                 "cloud-init", "user_data", "InstanceId", "ami-", "DLAMI",
                 "spot", "AWS_REGION", "queued_resource", "gcloud")

    def test_server_holds_no_cloud_vocabulary(self):
        # code_only, because server.py's docstring explains at length what was
        # removed and naming it there is the point.
        src = code_only(ROOT / "server.py")
        for word in self.FORBIDDEN:
            with self.subTest(word=word):
                self.assertNotIn(word, src)

    def test_requirements_pull_no_cloud_sdk(self):
        for name in ("requirements.txt", "requirements-serving.txt"):
            with self.subTest(file=name):
                text = (ROOT / name).read_text()
                for word in ("boto3", "botocore", "google-cloud", "awscli"):
                    self.assertNotIn(word, text)

    def test_no_aws_credential_helper_survives(self):
        # save-aws-creds.sh handled live credentials and had a guard against
        # writing them into a git work tree. Deleted rather than left dormant:
        # credential-handling code with nothing to authenticate is a liability.
        self.assertFalse((ROOT / "save-aws-creds.sh").exists())
        self.assertNotIn(".aws_creds", (ROOT / "Makefile").read_text())

    def test_jax_is_installed_without_a_cuda_extra(self):
        # `jax[cuda13]` on a CPU rig would pull ~3 GB of CUDA wheels and, worse,
        # could give this rig a GPU backend — at which point every number it
        # produced would be mislabelled.
        text = (ROOT / "requirements-serving.txt").read_text()
        self.assertIn("\njax\n", text)
        self.assertNotIn("cuda", text.split("# Requires Python")[1])

    def test_registration_files_all_name_this_rig(self):
        stem = f"{ROOT.name}-management"
        mcp_json = json.loads((ROOT / ".mcp.json").read_text())
        self.assertEqual(list(mcp_json["mcpServers"]), [ROOT.name])
        self.assertIn(stem, mcp_json["mcpServers"][ROOT.name]["args"][0])
        self.assertEqual(
            mcp_json["mcpServers"][ROOT.name]["env"]["MCP_SERVER_NAME"], ROOT.name
        )

        plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
        self.assertEqual(plugin["name"], ROOT.name)
        self.assertEqual(list(plugin["mcpServers"]), [ROOT.name])
        self.assertIn(stem, plugin["mcpServers"][ROOT.name]["args"][0])

        codex = (ROOT / ".codex" / "config.toml").read_text()
        self.assertIn(f"[mcp_servers.{ROOT.name}]", codex)
        self.assertIn(f".claude/skills/{stem}/mcp/server.py", codex)

        local = json.loads((ROOT / ".claude" / "settings.local.json").read_text())
        self.assertEqual(local["enabledMcpjsonServers"], [ROOT.name])

    def test_the_registered_entry_point_exists(self):
        # The fork's actual failure was a path, not a name: plugin.json pointed
        # at a skill directory that did not exist, and the rig was
        # unregisterable rather than merely misregistered.
        mcp_json = json.loads((ROOT / ".mcp.json").read_text())
        self.assertTrue((ROOT / mcp_json["mcpServers"][ROOT.name]["args"][0]).is_file())

    def test_nothing_is_approval_gated(self):
        # Deliberate, and worth pinning so it is not "fixed" by analogy with the
        # siblings: gating an action with no consequence trains the operator to
        # click through the ones that have.
        self.assertNotIn("approval_mode", (ROOT / ".codex" / "config.toml").read_text())


class CpuConstraintTests(unittest.TestCase):
    """The dtype policy inverts the GPU siblings', and for a different reason."""

    def test_dtype_default_is_bfloat16(self):
        # NOT float16, which is what every Turing sibling resolves. On a CPU
        # there is no 16-bit float datapath of any kind, so fp16 buys nothing
        # that bf16 does not — and bf16 is what the checkpoint already ships,
        # so it is also the one that costs no load-time cast.
        self.assertEqual(server.DTYPE, "bfloat16")

    def test_float32_is_documented_as_unaffordable_rather_than_unconsidered(self):
        # The interesting claim is not "we use bf16", it is WHY the conversion
        # tax cannot be removed here the way the GPU rigs remove it. If that
        # arithmetic is not written down, the next person re-derives it.
        src = (ROOT / "server.py").read_text()
        self.assertIn("float32", src)
        self.assertEqual(server._WEIGHT_BYTES_DENSE * 2, 18_514_000_000)

    def test_kv_cache_is_auto_and_never_fp8(self):
        argv = server._serve_argv()
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "auto")
        self.assertNotIn("fp8", argv)

    def test_quant_mode_matches_the_dense_checkpoint(self):
        # QUANT_MODE is a claim about MODEL_NAME, not about the host. A w4a16
        # mode against a dense checkpoint loads garbage rather than failing.
        argv = server._serve_argv()
        self.assertEqual(argv[argv.index("--quant-mode") + 1], "fp16")
        self.assertNotIn("w4a16", server.MODEL_NAME)

    def test_no_tensor_parallel_or_device_flag_is_emitted(self):
        argv = " ".join(server._serve_argv())
        self.assertNotIn("tensor-parallel", argv)
        self.assertNotIn("--device", argv)

    def test_the_backend_is_forced_to_cpu(self):
        # This machine also hosts the GPU sibling rigs. A CUDA plugin in
        # site-packages would silently hand this rig a GPU, and the rig would
        # keep its name while measuring something else entirely.
        self.assertEqual(server.JAX_PLATFORMS, "cpu")
        self.assertEqual(server._serve_env()["JAX_PLATFORMS"], "cpu")

    def test_int8_lm_head_is_off_here(self):
        # Inverted from every GPU sibling, deliberately: it ADDS 0.403 GB to buy
        # a memory-bandwidth win, and the constraint here is resident bytes.
        self.assertFalse(server.INT8_LM_HEAD)
        self.assertGreater(server._INT8_HEAD_COST_BYTES, 0)
        self.assertGreater(
            server._weight_estimate(int8_lm_head=True),
            server._weight_estimate(int8_lm_head=False),
            "int8_lm_head must be modelled as a memory COST, not a saving",
        )

    def test_ple_bits_is_on_and_is_modelled_as_a_saving(self):
        self.assertEqual(server.PLE_BITS, 4)
        self.assertLess(server._weight_estimate(4), server._weight_estimate(0))

    def test_binding_is_loopback_by_default(self):
        # Unlike every cloud sibling, which binds 0.0.0.0 because reaching the
        # box from elsewhere is the point there. This server has no auth.
        self.assertEqual(server.JAX_HOST, "127.0.0.1")
        argv = server._serve_argv()
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")


class HostCapacityTests(HostFactsMixin, unittest.TestCase):
    """The arithmetic that has to happen BEFORE a load.

    Exceeding a cloud quota is refused at the API. Exceeding host RAM is
    accepted and paid for in swap, and a thrashing serve is indistinguishable
    from a loading one — so this is the one check with no analogue in the
    siblings' failure model.
    """

    def test_weight_estimate_reproduces_the_measured_table(self):
        # gpu-jax-g5g-2b, 2026-08-26. Properties of the checkpoint and the
        # levers, not of the device, so they carry to this rig.
        self.assertEqual(server._weight_estimate(0, False), 9_257_000_000)
        self.assertEqual(server._weight_estimate(8, False), 6_927_000_000)
        self.assertEqual(server._weight_estimate(4, False), 5_752_000_000)
        self.assertEqual(server._weight_estimate(0, True), 9_660_000_000)
        self.assertEqual(server._weight_estimate(4, True), 6_155_000_000)

    def test_it_fits_when_it_fits(self):
        self.withFacts(Facts(ram_available=12 * GB, swap_free=0))
        out = run(server.check_host_capacity())
        self.assertIn("✅", out)
        self.assertIn("No swap needed", out)

    def test_swap_is_reported_as_working_but_not_healthy(self):
        # The verdict has to distinguish these, because they are different
        # findings: one is a configuration, the other is a benchmark you must
        # not record.
        self.withFacts(Facts(ram_available=2 * GB, swap_free=16 * GB))
        out = run(server.check_host_capacity())
        self.assertIn("⚠️", out)
        self.assertIn("out of swap", out)
        self.assertIn("do not record a benchmark", out)

    def test_it_says_no_when_neither_ram_nor_swap_can_hold_it(self):
        self.withFacts(Facts(ram_available=2 * GB, swap_free=0))
        out = run(server.check_host_capacity())
        self.assertIn("❌", out)
        # The remedies must be ordered cheapest-first and must actually be the
        # levers this rig has.
        self.assertIn("PLE_BITS=4", out)
        self.assertIn("INT8_LM_HEAD", out)
        self.assertIn("MAX_MODEL_LEN", out)

    def test_a_swapless_host_gets_the_btrfs_incident(self):
        """MEASURED 2026-08-29, and the error names the syscall not the cause.

        `fallocate` + `mkswap` + `swapon` — the remedy the parent rig renders in
        cloud-init — fails on a btrfs root with a bare `Invalid argument`. The
        real reason (`swapfile must not be copy-on-write`) reaches only dmesg,
        so the failure reads as a broken script rather than a filesystem that
        will not host that file.
        """
        self.withFacts(Facts(ram_available=12 * GB, swap_free=0, swap_total=0))
        out = run(server.check_host_capacity())
        self.assertIn("btrfs filesystem mkswapfile", out)
        self.assertIn("copy-on-write", out)

    def test_start_refuses_rather_than_thrashing(self):
        # Starting anyway would not fail cleanly. This is the one place the rig
        # takes a decision away from the operator, so it is worth pinning.
        self.withFacts(Facts(ram_available=1 * GB, swap_free=0))
        out = run(server.start_jax_server())
        self.assertIn("❌", out)
        self.assertIn("check_host_capacity", out)

    def test_available_not_free_is_what_is_read(self):
        # MemFree on a warm machine is near zero because the page cache holds
        # the rest; quoting it is how you conclude a 15 GB host has 0.5 GB.
        src = (ROOT / "server.py").read_text()
        body = src.split("def _host_facts")[1].split("\ndef ")[0]
        self.assertIn("MemAvailable", body)
        self.assertNotIn('"MemFree"', body)


class ProcessLifecycleTests(unittest.TestCase):
    """There is no instance, so the process IS the lifecycle."""

    def test_serve_argv_is_a_list_never_a_shell_string(self):
        # The monorepo rule: every subprocess call takes a list, never
        # shell=True. A serve command assembled as a string is the one place a
        # model id with a space in it becomes an injection.
        self.assertIsInstance(server._serve_argv(), list)
        self.assertNotIn("shell=True", code_only(ROOT / "server.py"))

    def test_serve_argv_points_at_the_resolved_payload(self):
        argv = server._serve_argv()
        self.assertTrue(argv[1].endswith("jax_openai_server.py"))
        self.assertTrue(Path(argv[1]).is_file())

    def test_pid_is_verified_against_the_command_line(self):
        """A recycled pid is how a process manager calls a dead service healthy.

        Not hypothetical on a developer box, where pid reuse is fast. The
        pidfile is written by us and the number is checked against
        /proc/<pid>/cmdline before it is believed.
        """
        saved = server.STATE_DIR
        server.STATE_DIR = str(Path(self.enterContext(_tempdir())))
        self.addCleanup(lambda: setattr(server, "STATE_DIR", saved))
        # pid 1 exists and is certainly not our serving process.
        with open(server._pidfile(), "w") as fh:
            fh.write("1")
        self.assertIsNone(server._read_pid())
        # A pid that does not exist at all must also be None, not an exception.
        with open(server._pidfile(), "w") as fh:
            fh.write("999999999")
        self.assertIsNone(server._read_pid())

    def test_endpoint_normalises_a_wildcard_bind_to_a_loopback_dial(self):
        # 0.0.0.0 is a bind address, not a destination. Dialling it is the one
        # way a local endpoint actually goes wrong.
        saved = server.JAX_HOST
        self.addCleanup(lambda: setattr(server, "JAX_HOST", saved))
        for bind in ("0.0.0.0", "::", ""):
            with self.subTest(bind=bind):
                server.JAX_HOST = bind
                self.assertIn("127.0.0.1", server._endpoint_base())
        server.JAX_HOST = "::1"
        self.assertEqual(server._endpoint_base(), f"http://[::1]:{server.JAX_PORT}/v1")

    def test_the_serve_outlives_the_agent(self):
        """MEASURED 2026-08-29: asyncio kills the child, start_new_session or not.

        The first version of start_jax_server used
        asyncio.create_subprocess_exec, like every other subprocess call in this
        monorepo. The serve died about a second after it started, with the
        device-policy banner in the log and NOTHING after it -- no traceback, no
        exit message. asyncio's subprocess transport kills its child when the
        event loop is deallocated, and start_new_session does not save it,
        because the kill is a direct kill() on the pid rather than a signal to a
        process group.

        For a model that takes minutes to load, a daemon that lives exactly as
        long as the tool call that started it is indistinguishable from a crash
        during loading -- which is why this is worth a test and not a comment.
        """
        body = code_only(ROOT / "server.py").split(
            "async def start_jax_server")[1].split("@mcp.tool")[0]
        self.assertIn("start_new_session=True", body)
        self.assertIn("subprocess.Popen(", body)
        self.assertNotIn("create_subprocess_exec", body)

    def test_the_asyncio_helper_is_still_used_for_everything_else(self):
        # The exception above is narrow: one daemon launch. Short-lived commands
        # stay on the asyncio helper, so the monorepo rule still describes the
        # code -- and Popen must not spread by copy-paste.
        src = code_only(ROOT / "server.py")
        self.assertEqual(src.count("subprocess.Popen("), 1)
        self.assertIn("asyncio.create_subprocess_exec", src)

    def test_stop_escalates_and_does_not_leave_a_stale_pidfile(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("async def stop_jax_server")[1].split("@mcp.tool")[0]
        self.assertIn("SIGTERM", body)
        self.assertIn("SIGKILL", body)
        self.assertIn("_clear_pidfile()", body)

    def test_rig_name_reaches_the_serving_process(self):
        # The metrics `rig` label is the only thing separating two rigs that
        # serve the same checkpoint, because the series names are identical.
        self.assertEqual(server._serve_env()["RIG_NAME"], server.RIG_NAME)

    def test_the_cache_dir_is_expanded_before_it_is_handed_over(self):
        # tpu.env spells it `~/.cache/...` and dotenv does not expand a tilde,
        # so a literal "~" would become a directory of that name in the cwd.
        self.assertFalse(server.JAX_COMPILATION_CACHE_DIR.startswith("~"))
        self.assertFalse(server._serve_env()["JAX_COMPILATION_CACHE_DIR"].startswith("~"))

    def test_the_documented_command_is_the_command_that_runs(self):
        """get_deployment_config drifted from the launch tool on the parent rig.

        It printed VolumeSize=200 while the launch created 100. A copy-pasteable
        repro that provisions something different from the tool it documents is
        how a manual reproduction fails to reproduce, so both render from one
        place — asserted here by checking the tool's output actually contains
        every token of _serve_argv().
        """
        shown = run(server.get_serve_command())
        for token in server._serve_argv():
            with self.subTest(token=token):
                self.assertIn(token, shown)

    def test_make_serve_shells_out_rather_than_copying_the_argv(self):
        # Same drift hazard, one layer out.
        makefile = (ROOT / "Makefile").read_text()
        serve = makefile.split("\nserve:")[1].split("\n\n")[0]
        self.assertIn("_serve_argv()", serve)
        self.assertNotIn("--ple-bits", serve)


def _tempdir():
    import tempfile
    return tempfile.TemporaryDirectory()


class DegeneracyGuardTests(unittest.TestCase):
    """The serving stack counted a token loop as status="success"."""

    @staticmethod
    def _looks_degenerate():
        # jax_openai_server imports jax at module scope; exec just this one pure
        # function instead of skipping the test.
        src = (ROOT / "jax_openai_server.py").read_text()
        body = src.split("def looks_degenerate")[1].split("\ndef _record")[0]
        ns = {}
        exec("def looks_degenerate" + body, ns)  # pure function from our own repo
        return ns["looks_degenerate"]

    def test_catches_both_degenerate_shapes_observed_on_hardware(self):
        f = self._looks_degenerate()
        # MEASURED on the G5g parent 2026-08-23 at 2,615-3,515 prompt tokens.
        self.assertTrue(f("The" * 40))
        # MEASURED on the vLLM sibling and recorded in the monorepo CLAUDE.md.
        self.assertTrue(f(": ok" * 20))
        self.assertTrue(f("ok " * 30))

    def test_does_not_fire_on_good_output(self):
        f = self._looks_degenerate()
        for good in (
            "The quick brown fox repeatedly jumps over a lazy dog.",
            "The three primary colours are: 1. Red 2. Yellow 3. Blue",
            "391",
            "Le chat est sur la table.",
            "yes yes yes yes yes but actually the answer depends on the context here",
            "",
        ):
            with self.subTest(text=good[:30]):
                self.assertFalse(f(good))

    def test_short_replies_are_not_judged(self):
        self.assertFalse(self._looks_degenerate()("ok " * 10))

    def test_counter_is_exposed_and_does_not_change_status(self):
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertIn("tpu_jax_degenerate_responses_total", text)
        self.assertIn('METRICS["successful_requests"] += 1', text)


class QuantKnobTests(unittest.TestCase):
    """The engine always supported these; a rig can fail to reach them."""

    def _argv(self, **env):
        saved = {k: getattr(server, k) for k in env}
        for k, v in env.items():
            setattr(server, k, v)
        try:
            return " ".join(server._serve_argv())
        finally:
            for k, v in saved.items():
                setattr(server, k, v)

    def test_ple_bits_is_always_emitted(self):
        # Emitted even at 0, so the serving command records the choice instead
        # of deferring to the server's own default.
        self.assertIn("--ple-bits 0", self._argv(PLE_BITS=0))
        self.assertIn("--ple-bits 4", self._argv(PLE_BITS=4))

    def test_int8_lm_head_is_a_flag_not_a_value(self):
        self.assertIn("--int8-lm-head", self._argv(INT8_LM_HEAD=True))
        self.assertNotIn("--int8-lm-head", self._argv(INT8_LM_HEAD=False))

    def test_prefill_chunk_size_is_omitted_when_unset(self):
        # Unset must mean one-shot prefill, not a chunk size of 0.
        self.assertNotIn("--prefill-chunk-size", self._argv(PREFILL_CHUNK_SIZE=""))
        self.assertIn("--prefill-chunk-size 1024", self._argv(PREFILL_CHUNK_SIZE="1024"))

    def test_serving_process_accepts_every_flag_the_rig_emits(self):
        # The rig and the server are separate files and drifted before: the
        # server grew --ple-bits and _serve_argv never learned to pass it.
        text = (ROOT / "jax_openai_server.py").read_text()
        argv = self._argv(PLE_BITS=4, INT8_LM_HEAD=True, PREFILL_CHUNK_SIZE="1024")
        for token in argv.split():
            if token.startswith("--"):
                self.assertIn(f'"{token}"', text, f"{token} is emitted but not accepted")

    def test_load_engine_defaults_are_not_the_tpu_rigs(self):
        text = (ROOT / "jax_openai_server.py").read_text()
        signature = text.split("def load_engine(", 1)[1].split(")", 1)[0]
        self.assertNotIn('"bf16"', signature)
        self.assertNotIn('"w4a16"', signature)


class MetricsParsingTests(unittest.TestCase):
    """_parse_prom is pure, so the interesting parts pin offline."""

    EXPOSITION = "\n".join([
        "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
        "# TYPE tpu_jax_precision_info gauge",
        'tpu_jax_precision_info{model="google/gemma-4-E2B-it",compute_dtype="bfloat16",'
        'quant_mode="fp16",kv_cache_dtype="bfloat16",kv_cache_requested="auto",'
        'ple_bits="4",int8_lm_head="false",rig="local-jax-cpu-2b"} 1',
        "",
        'tpu_jax_requests_total{model="google/gemma-4-E2B-it",status="success"} 3',
        'tpu_jax_requests_total{model="google/gemma-4-E2B-it",status="failed"} 0',
        'tpu_jax_decode_tokens_per_second{model="google/gemma-4-E2B-it"} 1.7',
        'tpu_jax_host_rss_bytes{model="google/gemma-4-E2B-it"} 6155000000',
        "",
    ])

    def setUp(self):
        self.samples, self.precision, self.model = server._parse_prom(self.EXPOSITION)

    def test_served_checkpoint_survives_parsing(self):
        # The model label is dropped from every row as noise; if it were not
        # hoisted out first, a metrics transcript could not be checked against
        # MODEL_NAME -- which is the whole point of quoting one.
        self.assertEqual(self.model, "google/gemma-4-E2B-it")

    def test_precision_is_split_out_not_rendered_as_a_number(self):
        # The info series carries its payload in labels and a constant value of
        # 1, so leaving it among the samples would print a meaningless "1" row.
        self.assertNotIn("tpu_jax_precision_info", "".join(self.samples))
        self.assertEqual(self.precision["compute_dtype"], "bfloat16")
        self.assertEqual(self.precision["kv_cache_requested"], "auto")
        self.assertEqual(self.precision["rig"], "local-jax-cpu-2b")

    def test_distinguishing_labels_are_kept(self):
        self.assertEqual(self.samples['tpu_jax_requests_total{status="success"}'], 3.0)
        self.assertEqual(self.samples['tpu_jax_requests_total{status="failed"}'], 0.0)
        self.assertEqual(self.samples["tpu_jax_decode_tokens_per_second"], 1.7)
        self.assertEqual(self.samples["tpu_jax_host_rss_bytes"], 6155000000.0)

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertFalse([k for k in self.samples if k.startswith("#")])

    def test_empty_exposition_yields_nothing(self):
        samples, precision, model = server._parse_prom("")
        self.assertEqual((samples, precision, model), ({}, {}, None))


class HostMemoryReportingTests(unittest.TestCase):
    """A CPU device has no allocator, and 0 is the wrong way to say so."""

    def test_hbm_is_absent_rather_than_zero_without_an_allocator(self):
        """On an accelerator `hbm_bytes_in_use == 0` means the allocator holds
        nothing, which is a symptom. Here it would mean there is no allocator,
        which is not — and it would make the host-RSS series look redundant."""
        src = (ROOT / "jax_openai_server.py").read_text()
        self.assertIn('if mem.get("has_device_allocator"):', src)
        exporter = src.split("def metrics")[1] if "def metrics" in src else src
        self.assertIn("tpu_jax_host_rss_bytes", exporter)

    def test_the_engine_distinguishes_absent_from_empty(self):
        src = (ROOT / "jax_engine.py").read_text()
        body = src.split("def memory_stats")[1].split("\n    def ")[0]
        self.assertIn("has_device_allocator", body)
        self.assertIn("host_rss_bytes", body)
        # memory_stats() can also return None on a device that HAS the accessor.
        self.assertIn("stats = stats or {}", body)

    def test_rss_is_read_without_a_new_dependency(self):
        # A serving dependency that exists to read one number is a dependency
        # that can be missing exactly when you are trying to find out why memory
        # ran out.
        src = (ROOT / "jax_engine.py").read_text()
        self.assertIn("/proc/self/statm", src)
        self.assertNotIn("psutil", code_only(ROOT / "jax_engine.py"))
        self.assertNotIn("psutil", (ROOT / "requirements-serving.txt").read_text())

    def test_rss_reads_something_real_on_this_process(self):
        from jax_engine import _host_rss_bytes
        self.assertGreater(_host_rss_bytes(), 1_000_000)


class ServedPrecisionTests(unittest.TestCase):
    """The precision the DEVICE resolved, never the one that was requested."""

    def test_health_does_not_hardcode_bfloat16(self):
        # jax_openai_server used to report activations="bfloat16" and
        # weights="bf16" unconditionally, inherited from the TPU rig. On this
        # rig bfloat16 happens to be right, which makes a hardcoded value MORE
        # dangerous rather than less: it would agree with reality until someone
        # set JAX_E_COMPUTE_DTYPE and then quietly stop.
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertNotIn('"activations": "bfloat16"', text)
        self.assertIn("precision_info()", text)

    def test_load_banner_does_not_claim_w4a16(self):
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertNotIn("Loading W4A16 QAT weights", text)

    def test_engine_reports_resolved_not_requested(self):
        text = (ROOT / "jax_engine.py").read_text()
        self.assertIn("def precision_info", text)
        for key in ("compute_dtype", "kv_cache_dtype", "kv_cache_requested", "quant_mode"):
            self.assertIn(f'"{key}"', text)

    def test_pallas_interpret_is_reported_because_it_is_a_simulator(self):
        """The one silent failure this rig has that the GPU siblings do not.

        Pallas has no CPU backend, so the fused W4A16 kernel AUTO-ENABLES
        interpret mode here rather than being refused. It then produces correct
        numbers at a speed that means nothing. The banner is the only warning.
        """
        src = (ROOT / "ports" / "gemma4" / "jax_e_model.py").read_text()
        self.assertIn("pallas_interpret=%s", src)
        self.assertIn('"0" if PLATFORM in ("tpu", "gpu", "cuda") else "1"', src)


class PayloadTests(unittest.TestCase):
    """No tarball and no deploy — but the digest still answers a real question."""

    def test_payload_files_exist_and_are_found(self):
        root = Path(server._payload_root())
        for rel in server._PAYLOAD_FILES:
            with self.subTest(path=rel):
                self.assertTrue((root / rel).is_file())

    def test_nothing_is_shipped_anywhere(self):
        src = (ROOT / "server.py").read_text()
        for gone in ("_payload_tar_b64", "tarfile", "base64"):
            with self.subTest(symbol=gone):
                self.assertNotIn(gone, src)

    def test_the_digest_is_deterministic_and_content_addressed(self):
        first, second = server._payload_digest(), server._payload_digest()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertRegex(first, r"^[0-9a-f]{12}$")

    def test_the_server_computes_the_same_digest_from_the_files_it_runs(self):
        """This is what makes the build id mean anything with no deploy step.

        jax_openai_server falls back to hashing the payload in place when no
        PAYLOAD_SHA stamp is present, which is always the case here. If the two
        implementations ever diverge, every health check reports a phantom
        mismatch — so pin that they hash the same list in the same order.
        """
        src = (ROOT / "jax_openai_server.py").read_text()
        body = src.split("def _read_build_id")[1].split("\n\n\n")[0]
        for rel in server._PAYLOAD_FILES:
            with self.subTest(path=rel):
                self.assertIn(rel, body)
        self.assertIn("sorted(", body)
        self.assertIn("[:12]", body)


class LintCoverageTests(unittest.TestCase):
    """`make lint` lints a HARDCODED list, so a new module is silently unlinted.

    That is not hypothetical: profile_decode.py sat outside the list and was red
    for a day. ports/ is excluded on purpose -- ruff's UP006/UP045 would rewrite
    its Dict/Optional annotations, which the monorepo CLAUDE.md forbids and which
    would drift it from the copy tpu-jax-v5e1-2b shares.
    """

    def _ruff_line(self):
        for line in (ROOT / "Makefile").read_text().splitlines():
            if line.strip().startswith("ruff check"):
                return line
        self.fail("no `ruff check` line in the Makefile")

    def test_every_top_level_module_is_on_the_ruff_line(self):
        # Checked against the RUFF LINE, not the whole Makefile. The old version
        # searched the entire file, so make-medium.py passed vacuously on the
        # strength of appearing in the `medium` recipe while never being linted.
        listed = self._ruff_line()
        modules = sorted(p.name for p in ROOT.glob("*.py"))
        self.assertTrue(modules, "no top-level modules found")
        for m in modules:
            with self.subTest(module=m):
                self.assertIn(m, listed, f"{m} is not on the `ruff check` line")

    def test_the_lint_list_names_no_module_that_is_gone(self):
        # The other direction, which is how a fork leaves `make lint` failing on
        # a file it deleted.
        listed = self._ruff_line().replace("ruff check", "").split()
        for entry in listed:
            if entry.endswith(".py"):
                with self.subTest(module=entry):
                    self.assertTrue((ROOT / entry).is_file(), f"{entry} does not exist")


class CompilationCacheDirTests(unittest.TestCase):
    """JAX_COMPILATION_CACHE_DIR must survive the port's import.

    MEASURED on the G5g parent 2026-08-27: the unit set the variable, the
    process had it, and the configured directory stayed EMPTY while 447 files
    accumulated under the fallback. ports/gemma4/jax_e_model.py set the path
    unconditionally at import, and jax_openai_server imports it (via jax_engine)
    AFTER resolving the same variable -- so the port silently won, every start.

    Nothing failed and nothing logged. This rig's default is the port's own
    fallback path, which makes the two agree by construction; the tests stay
    because "they agree today" is not the same as "they cannot disagree".
    """

    def _source(self, rel):
        return (ROOT / rel).read_text()

    def test_the_port_honours_the_env_var(self):
        src = self._source("ports/gemma4/jax_e_model.py")
        self.assertIn('os.environ.get("JAX_COMPILATION_CACHE_DIR")', src)

    def test_the_port_does_not_hardcode_the_fallback_as_the_only_path(self):
        src = self._source("ports/gemma4/jax_e_model.py")
        self.assertNotIn(
            '_cache_dir = os.path.expanduser("~/.cache/jax_compilation_cache")', src
        )

    def test_both_modules_resolve_the_cache_dir_the_same_way(self):
        # They run in one process and the later import wins, so agreeing is the
        # only way the result does not depend on import order.
        for rel in ("ports/gemma4/jax_e_model.py", "jax_openai_server.py"):
            with self.subTest(module=rel):
                src = self._source(rel)
                self.assertIn('os.environ.get("JAX_COMPILATION_CACHE_DIR")', src)
                self.assertIn('"~/.cache/jax_compilation_cache"', src)

    def test_the_serving_process_actually_gets_the_variable(self):
        # The env var is only worth honouring if the launcher sets it.
        self.assertEqual(
            server._serve_env()["JAX_COMPILATION_CACHE_DIR"],
            server.JAX_COMPILATION_CACHE_DIR,
        )

    def test_the_cache_is_not_ephemeral_here(self):
        # The S3 sync, the systemd timer and the operator-supplied bucket URI
        # exist on the cloud rigs because their cache dies with the instance.
        # Porting them here would back up a directory that already persists.
        src = (ROOT / "server.py").read_text()
        self.assertNotIn("JAX_CACHE_S3_URI", src)
        self.assertNotIn("JAX_CACHE_SYNC_MINUTES", src)


class RepoHygieneTests(BashSyntaxMixin, unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh"):
            with self.subTest(script=script):
                self.assertScriptParses(ROOT / script)

    def test_tpu_env_agrees_with_server_defaults(self):
        # The directory name is a claim about tpu.env (NAMING.md). This asserts
        # the env file and the server actually agree, so the claim stays true.
        values = tpu_env()
        for key in ("MODEL_NAME", "DTYPE", "KV_CACHE_DTYPE", "QUANT_MODE",
                    "PREFILL_CHUNK_SIZE", "JAX_PLATFORMS", "JAX_HOST",
                    "XLA_CPU_THREADS"):
            with self.subTest(key=key):
                self.assertEqual(values[key], getattr(server, key))
        for key in ("JAX_PORT", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "PLE_BITS"):
            with self.subTest(key=key):
                self.assertEqual(int(values[key]), getattr(server, key))
        # Booleans are a string in the env file and a bool in the server, so
        # they need their own comparison rather than being skipped -- on the
        # parent rig INT8_LM_HEAD flipped and nothing asserted the two agreed.
        self.assertEqual(
            values["INT8_LM_HEAD"].lower() in ("1", "true", "yes"), server.INT8_LM_HEAD
        )
        # The cache dir needs expanding on both sides before comparison, which
        # is the whole reason server.py expands the RESOLVED value.
        self.assertEqual(
            os.path.expanduser(values["JAX_COMPILATION_CACHE_DIR"]),
            server.JAX_COMPILATION_CACHE_DIR,
        )

    def test_no_dead_config_survives_the_fork(self):
        # Two forks deep from a vLLM rig, and each hop left keys describing
        # hardware the rig does not have. A leftover key reads as live config.
        text = (ROOT / "tpu.env").read_text()
        dead = ("VLLM_", "TORCH_CUDA", "INSTANCE_TYPE", "AWS_", "DLAMI",
                "ROOT_VOLUME", "SERVICE_NAME", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "TENSOR_PARALLEL_SIZE", "JAX_CACHE_S3_URI")
        live = [
            ln for ln in text.splitlines()
            if ln and not ln.startswith("#") and ln.split("=")[0].startswith(dead)
        ]
        self.assertEqual(live, [])

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_the_hardware_slot_is_honest(self):
        # `cpu` names whatever machine the rig is checked out on, unlike every
        # sibling's slot, which names a SKU the rig provisions. That has to be
        # written down, or two runs on two machines get compared.
        self.assertEqual(ROOT.name.split("-")[2], "cpu")
        self.assertEqual(ROOT.name.split("-")[0], "local")
        self.assertIn("THE HOST IS NOT PART OF THE RIG", (ROOT / "tpu.env").read_text())

    def test_skill_is_complete_in_both_copies(self):
        # SKILL.md is a hand-written SOURCE file, but refresh_skill.py only
        # regenerates the mcp/ files beside it. So `rm -rf .claude/skills`
        # destroys it and `make skill` does not bring it back — which is exactly
        # what happened during the t4g->g5g rename. Guard both copies.
        stem = f"{ROOT.name}-management"
        for prefix in (f".claude/skills/{stem}", f"skills/{stem}"):
            with self.subTest(copy=prefix):
                skill = ROOT / prefix / "SKILL.md"
                self.assertTrue(skill.is_file(), f"{prefix}/SKILL.md is missing")
                self.assertIn(
                    f"name: {stem}", skill.read_text(),
                    f"{prefix}/SKILL.md has a stale name",
                )
            # subTest keeps the loop going past a failure, so the wiped-directory
            # case this test exists to catch would reach filecmp.cmp and bury the
            # one useful message under a FileNotFoundError per source file.
            if not (ROOT / prefix / "mcp").is_dir():
                continue
            for source in SKILL_SOURCES:
                snapshot = ROOT / prefix / "mcp" / source
                with self.subTest(copy=prefix, source=source):
                    self.assertTrue(snapshot.is_file(), f"{snapshot} is missing — run `make skill`")
                    self.assertTrue(
                        filecmp.cmp(ROOT / source, snapshot, shallow=False),
                        f"{prefix}/mcp/{source} is stale — run `make skill`",
                    )

    def test_serving_requirements_match_the_mirror_file(self):
        listed = {
            ln.strip() for ln in (ROOT / "requirements-serving.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        self.assertTrue(set(server._SERVING_REQUIREMENTS) <= listed | {"jax"})
        for pkg in server._SERVING_REQUIREMENTS:
            with self.subTest(package=pkg):
                self.assertIn(pkg, listed)

    def test_profiling_requirements_match_the_mirror_file(self):
        listed = {
            ln.strip() for ln in (ROOT / "requirements-profiling.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        self.assertEqual(set(server._PROFILING_REQUIREMENTS), listed)

    def test_the_profiler_is_never_on_the_serving_list(self):
        # A failure in requirements-serving.txt must stop a serve; a failure in
        # the profiling list must not.
        serving = (ROOT / "requirements-serving.txt").read_text()
        for pkg in server._PROFILING_REQUIREMENTS:
            with self.subTest(package=pkg):
                self.assertNotIn(f"\n{pkg}\n", serving)


class ObservabilityTests(unittest.TestCase):
    """Pin the traceability machinery inherited from the 2026-08-25 rework.

    Every assertion here corresponds to a way the parent rig previously
    destroyed its own evidence: dropped INFO logs, tracebacks discarded on 500s,
    a request id that reached nothing, and padding computed and thrown away.
    None of it is cloud-specific, and all of it matters more here rather than
    less — this rig has no journal, no instance console and no second machine to
    compare against, so the logfile is the entire record.
    """

    SERVER_SRC = None
    ENGINE_SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SERVER_SRC = (ROOT / "jax_openai_server.py").read_text()
        cls.ENGINE_SRC = (ROOT / "jax_engine.py").read_text()

    # ----------------------------------------------------------- logging

    def test_root_logging_is_configured_before_the_engine_import(self):
        """Ordering is the whole fix, not the basicConfig call on its own.

        ports.gemma4.jax_e_model logs its device-policy banner at IMPORT time and
        is imported via jax_engine, so configuring after that import would
        silence the one line naming the resolved compute dtype — and on this rig
        that same line is the only place `pallas_interpret=True` is announced.
        """
        src = self.SERVER_SRC
        config_at = src.index("logging.basicConfig(")
        engine_import_at = src.index("from jax_engine import")
        self.assertLess(
            config_at, engine_import_at,
            "logging.basicConfig must precede the jax_engine import, or the "
            "device-policy banner is emitted before any handler exists",
        )
        self.assertIn("force=True", src[config_at:engine_import_at])

    def test_uvicorn_does_not_configure_the_root_logger(self):
        """The premise of the fix, asserted rather than assumed."""
        import logging
        import logging.config

        try:
            from uvicorn.config import LOGGING_CONFIG
        except ImportError:
            self.skipTest("uvicorn is a serving dependency, absent here")
        root = logging.getLogger()
        saved = list(root.handlers), root.level
        try:
            root.handlers = []
            logging.config.dictConfig(LOGGING_CONFIG)
            self.assertEqual(
                root.handlers, [],
                "uvicorn now configures the root logger; revisit basicConfig",
            )
        finally:
            root.handlers, root.level = saved

    def test_startup_reports_through_the_logger_not_print(self):
        body = self.SERVER_SRC.split("def load_engine")[1].split("\ndef _eos_ids")[0]
        self.assertNotIn("print(", body)
        self.assertIn("logger.info", body)

    # -------------------------------------------------------- request path

    def test_failures_are_logged_with_the_traceback_and_the_request_id(self):
        self.assertEqual(self.SERVER_SRC.count("logger.exception("), 3,
                         "expected both handlers plus the streaming generator")
        self.assertIn('detail=f"[{req_id}] {exc}"', self.SERVER_SRC)

    def test_streaming_failures_are_counted_and_not_silent(self):
        """The generator runs after the handler returned, outside its try."""
        stream = self.SERVER_SRC.split("def _sse_stream")[1].split('@app.get("/health")')[0]
        self.assertIn('METRICS["failed_requests"] += 1', stream)
        self.assertIn('finish="error"', stream)

    def test_request_id_is_echoed_in_a_header(self):
        self.assertIn('response.headers["X-Request-Id"] = req_id', self.SERVER_SRC)
        self.assertEqual(self.SERVER_SRC.count('"X-Request-Id": req_id'), 2,
                         "both streaming responses must carry it too")

    def test_one_log_line_per_request_carries_the_shape(self):
        record = self.SERVER_SRC.split("def _record")[1].split("\ndef _sse_stream")[0]
        for field in ("id=%s", "bucket=%d", "pad=%d", "cold=%s", "finish=%s"):
            with self.subTest(field=field):
                self.assertIn(field, record)

    # ------------------------------------------------------------- padding

    def test_generation_stats_carries_the_variable_behind_the_eviction_bug(self):
        """pad_len predicted the failure 14/14 and was being discarded."""
        for field in ("bucket_size", "pad_len", "cold_shape",
                      "prefill_chunked", "max_new_tokens_clamped"):
            with self.subTest(field=field):
                self.assertIn(f"{field}:", self.ENGINE_SRC)
        self.assertIn("pad_len = bucket_s - prompt_len", self.ENGINE_SRC)

    def test_padding_at_or_past_the_window_is_warned_about(self):
        self.assertIn("the pre-fix eviction", self.ENGINE_SRC)
        self.assertIn("pad_len >= int(win)", self.ENGINE_SRC)

    def test_degenerate_output_is_logged_with_its_padding(self):
        record = self.SERVER_SRC.split("def _record")[1].split("\ndef _sse_stream")[0]
        self.assertIn("DEGENERATE OUTPUT", record)
        self.assertIn("logger.error", record)

    def test_pad_metrics_are_exported(self):
        for series in ("tpu_jax_last_pad_tokens", "tpu_jax_max_pad_tokens",
                       "tpu_jax_last_bucket_size", "tpu_jax_cold_requests_total",
                       "tpu_jax_decode_seconds_total"):
            with self.subTest(series=series):
                self.assertIn(series, self.SERVER_SRC)

    def test_metrics_carry_a_rig_label(self):
        # The series names are byte-identical across the JAX rigs ON PURPOSE:
        # both benchmark reports compare on tpu_jax_decode_tokens_per_second BY
        # NAME, so the label is what separates them and a rename would break
        # continuity for no gain. Keeping a `tpu_jax_` prefix on a rig with no
        # TPU is the price, and it is the right one.
        self.assertIn('f\'rig="{RIG_NAME}"\'', self.SERVER_SRC)
        self.assertIn('RIG_NAME = os.environ.get("RIG_NAME"', self.SERVER_SRC)
        self.assertIn('env["RIG_NAME"] = RIG_NAME', (ROOT / "server.py").read_text())

    # ---------------------------------------------------- silent fallbacks

    def test_the_two_silent_fallbacks_now_warn(self):
        self.assertIn("to ONE-SHOT prefill for this shape", self.ENGINE_SRC)
        self.assertIn("max_new_tokens clamped", self.ENGINE_SRC)

    # ------------------------------------------------------ build identity

    def test_the_server_reports_a_build_id(self):
        self.assertIn('"build_id": BUILD_ID', self.SERVER_SRC)
        self.assertIn('response.headers["X-Build-Id"] = BUILD_ID', self.SERVER_SRC)

    def test_start_reports_the_root_it_resolved(self):
        # _payload_root() silently picks between the working tree and the skill
        # snapshot. There is no deploy here, so the question is not "is this
        # stale?" but "which copy am I running?" — and it has the same answer.
        src = (ROOT / "server.py").read_text()
        body = src.split("async def start_jax_server")[1].split("@mcp.tool")[0]
        self.assertIn("Payload root:", body)
        self.assertIn("build id", body.lower())

    # ------------------------------------------------------- health verdict

    def test_health_does_not_pass_on_a_merely_non_empty_reply(self):
        """The rule the tool itself used to break.

        A broken deploy answered ': ok: ok: ok…' on the vLLM sibling, and KV-ring
        eviction returned a token loop — both non-empty, both broken.
        """
        src = (ROOT / "server.py").read_text()
        body = src.split("async def verify_model_health")[1].split("@mcp.tool")[0]
        self.assertNotIn("and text.strip() else", body)
        self.assertIn("tpu_jax_degenerate_responses_total", body)
        self.assertIn("degenerate", body)

    def test_health_compares_the_served_build_against_the_local_one(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("async def verify_model_health")[1].split("@mcp.tool")[0]
        self.assertIn("DIFFERENT PAYLOAD", body)
        self.assertIn("_payload_digest()", body)

    def test_health_is_503_while_loading(self):
        # Which matters more here than anywhere else: the load is minutes long
        # and there is no instance state to consult while it runs.
        self.assertIn("response.status_code = 503", self.SERVER_SRC)

    def test_error_renderer_logs_the_traceback(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("def _error(exc")[1].split("@mcp.tool")[0]
        self.assertIn("logger.exception", body)


class QuantizerMemoryTests(unittest.TestCase):
    """The load-time quantizers must not allocate the destination on top of the
    source. Both bugs below were hard startup failures on a T4G 2026-08-26, and
    both allocations matched their tensor byte-for-byte.

    They matter here for a different reason than they did there. On a device
    with a budget the failure was RESOURCE_EXHAUSTED at load; on this rig the
    same double-residency is absorbed by swap, so it does not fail — it just
    makes a five-minute load into a twenty-minute one, with nothing in the log
    to say why. A bug that stops announcing itself is worth more tests, not
    fewer."""

    SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SRC = (ROOT / "ports" / "gemma4" / "jax_e_model.py").read_text()

    def _fn(self, name):
        body = self.SRC.split(f"def {name}")[1]
        return body.split("\ndef ")[0]

    def test_lm_head_quantizes_on_the_host_in_chunks(self):
        """262144 x 1536 x 4 B = 1.50 GiB, the exact allocation that failed."""
        body = self._fn("quantize_lm_head")
        self.assertIn('jax.devices("cpu")', body)
        self.assertIn("rows_per_chunk", body)
        # The upcast must apply to a CHUNK, never the whole table. Checked on
        # executable lines only -- the comment above the fix quotes the old
        # expression on purpose, and matching prose would pass vacuously.
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.lstrip().startswith("#"))
        self.assertIn("blk = blk.astype(jnp.float32)", code)
        self.assertNotIn("emb.astype(jnp.float32)", code)

    def test_lm_head_moves_the_table_host_side_before_slicing(self):
        body = self._fn("quantize_lm_head")
        move = body.index("emb_host = jax.device_put(emb, cpu)")
        first_slice = body.index("emb_host[start:start + rows_per_chunk]")
        self.assertLess(move, first_slice)

    def test_ple_releases_the_source_before_placing_the_copy(self):
        """262144 x 8960 x 1 B = 2.19 GiB, the exact allocation that failed."""
        body = self._fn("quantize_ple_table")
        self.assertIn("source.delete()", body)
        # Opt-in: .delete() invalidates the CALLER's array. A CPU test caught
        # this within seconds of the first version defaulting it to on.
        self.assertIn("if release_source:", body)
        engine = (ROOT / "jax_engine.py").read_text()
        self.assertIn("release_source=True", engine,
                      "load() must opt in, or the fix does nothing in production")
        delete_at = body.index("source.delete()")
        place_at = body.index("q_all = jax.device_put(q_all, home)")
        self.assertLess(delete_at, place_at,
                        "the source must be released BEFORE the copy is placed")

    def test_the_failing_allocations_match_their_tensors(self):
        self.assertAlmostEqual(262144 * 1536 * 4 / 2**30, 1.50, places=2)
        self.assertAlmostEqual(262144 * 8960 * 1 / 2**30, 2.19, places=2)


if __name__ == "__main__":
    unittest.main()
