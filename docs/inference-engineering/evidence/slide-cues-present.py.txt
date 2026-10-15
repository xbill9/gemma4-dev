"""Step through the talk: one cue per slide, with the live command for that slide.

Run it in the terminal you will show, next to the Google Slides deck:

    source stage/demo.env
    python3 stage/present.py              # from slide 1
    python3 stage/present.py --start 27   # pick up at the Strands section
    python3 stage/present.py --list       # print every cue and command, run nothing

Keys (type the letter, then Enter):
    Enter   run this slide's next command; with none left, go to the next slide
    n / p   next / previous slide          g N   go to slide N
    r       show the recorded run for the current command instead of running it
    s       skip the current command       l     show the slide's link again
    q       quit

Each slide header shows its link (#slide=id.<object id>) so the deck can be jumped to
from the terminal, the minutes planned so far, and the clock since slide 1. A
command whose backend is not configured, or that exits non-zero, offers its
recorded run from results/2026-10-05-stage/.
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DECK = "https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit"
# Google Slides object id of each slide, in deck order (slide N is SLIDE_IDS[N - 1]). Ids were
# fixed when each slide was made, so they no longer follow the slide numbers.
SLIDE_IDS = [
    "sl_01", "sl_02", "sl_03", "sl_04", "sl_05", "sl_prec", "sl_mem", "sl_06",
    "sl_07", "sl_08", "sl_09", "sl_10", "sl_11", "sl_12", "sl_smi", "sl_13",
    "sl_14", "sl_15", "sl_l4", "sl_16", "sl_arm", "sl_17", "sl_18", "sl_19",
    "sl_20", "sl_why", "sl_21", "sl_22", "sl_23", "sl_24", "sl_25", "sl_26",
    "sl_27", "sl_28", "sl_29", "sl_30",
]


def link(n: int) -> str:
    return f"{DECK}#slide=id.{SLIDE_IDS[n - 1]}"
RECORDED = "results/2026-10-05-stage"
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def ask(backend: str) -> dict:
    return {"cmd": f"python3 stage/ask.py {backend}", "needs": backend,
            "recorded": f"{RECORDED}/ask-{backend}.txt"}


# (slide, title, minutes, notes, commands)
CUES = [
    (1, "Gemma 4 Inference on AWS", 1.0,
     ["Six ways to serve one model on AWS, one Strands agent over all of them."], []),
    (2, "About me", 1.0, ["GDE Cloud and AI/ML, AWS Community Builder."], []),
    (3, "Where to find me", 0.5, ["Everything today is on GitHub and dev.to."], []),
    (4, "One Question, Six Ways to Answer It", 2.0,
     ["Managed API, managed endpoint, your GPU, your AWS chip.",
      "Grading and timing happen in code, never in the model."], []),
    (5, "The Six Backends", 2.0,
     ["g6 and SageMaker: same weights, same GPU type; VM against managed endpoint.",
      "inf2 and trn1: same image digest, two chips."], []),
    (6, "Precision: What You Trade", 2.5,
     ["Memory: fewer bits per weight, less memory, disk and bandwidth. E2B goes from 9.42 GiB in bf16 to 2.85 GiB with int4 weights and int4 tables.",
      "Quality: int4 built from Google's QAT weights stays within half a point of bf16; rounding int8 from bf16 lost 1.3.",
      "Native math: the L4 multiplies bf16, fp8 and int8; the T4G has no bf16 or fp8; int4 is widened to 16-bit everywhere here."], []),
    (7, "Memory and Bandwidth Across AWS", 2.0,
     ["Decode reads every weight once per token, so memory bandwidth sets the ceiling on tokens per second.",
      "L4 300 GB/s, T4G 320 (277 measured), Inferentia2 ~817, Trainium ~613: Neuron has 2.0-2.7x the L4's bandwidth.",
      "Neuron's 32 GB is 2 x 16 GiB, one per NeuronCore: a model splits across both cores or fits in 16 GiB.",
      "Yet the L4 decodes 3.6x faster: the engine (vLLM vs a hand port) decides how close you get to the ceiling."], []),
    (8, "Strategies", 0.5, ["Four habits that decided whether a run happened at all."], []),
    (9, "Quota Does Not Mean Availability", 2.5,
     ["Rehearsal: inf2 refused in five zones across us-east-2 and us-east-1, launched in us-west-2a.",
      "Quota lets you ask; the zone decides. Keep a second zone and a second region."],
     [{"cmd": "grep -E 'no capacity|launched|already up' stage/up.log", "recorded": f"{RECORDED}/up.log"}]),
    (10, "Debug With the Small Model", 2.0,
     ["E2B on Neuron is ready in about 2 minutes, the 26B in close to 4.",
      "Wiring breaks the same way at any size; the small model breaks faster."], []),
    (11, "Check Your Numbers, Verify Results", 2.0,
     ["Code counts, times and ranks; the model quotes.",
      "Grade the filter the agent sent, as well as the number it said."], []),
    (12, "Keep IAM Loose While You Build", 1.5,
     ["One instance profile, one SSM-only security group per region, no inbound ports.",
      "Tighten before anything faces users."], []),
    (13, "Managed: Bedrock and SageMaker", 0.5, [], []),
    (14, "Bedrock: Nothing to Start", 2.5,
     ["An IAM permission and an inference profile id.",
      "At temperature 0 Bedrock gave 9 different paragraphs in 10 repeats."],
     [ask("bedrock")]),
    (15, "SageMaker: You Bring the Container", 1.5,
     ["Three API calls: a model (container image, SM_VLLM_* settings, IAM role), an endpoint config (ml.g6.xlarge x 1), an endpoint.",
      "SageMaker starts the instance, the vLLM container pulls the weights from Hugging Face, and traffic starts after the health check.",
      "Billed per instance hour until the endpoint is deleted."], []),
    (16, "SageMaker: The Same vLLM, Managed", 3.0,
     ["About 9 minutes to InService on 2026-10-12, on the AWS vLLM 0.31.0 container.",
      "Same opening sentence as the g6 on slide 18; the rest differs, since the g6 runs vLLM 0.30."],
     [ask("sagemaker")]),
    (17, "GPUs on EC2", 0.5, [], []),
    (18, "g6.xlarge: vLLM 0.30 on an NVIDIA L4", 3.0,
     ["16 minutes launch to ready on 2026-10-12; the first zone had no capacity.",
      "Fastest decode here: 131.9 tok/s. Compare its paragraph with SageMaker's."],
     [ask("g6")]),
    (19, "NVIDIA L4: The GPU in g6 and SageMaker", 1.5,
     ["Ada Lovelace, compute capability 8.9: native bf16 and fp8, which the T4G next door lacks.",
      "24 GB and 300 GB/s: one request at a time, decode speed tracks the 300 GB/s."], []),
    (20, "g5g.2xlarge: Arm Host, NVIDIA T4G", 2.5,
     ["Patched vLLM for sm_75; weights read from a fresh EBS volume for about 530 s.",
      "23.7 minutes launch to ready: start it first."],
     [ask("g5g")]),
    (21, "Arm + CUDA: What Is Old Is New", 2.0,
     ["AWS calls G5g the first Arm-based instances in a major cloud with GPU acceleration.",
      "NVIDIA's RTX Spark puts Arm and CUDA in Windows laptops this fall: 20-core Grace, Blackwell RTX, up to 128 GB unified.",
      "The aarch64 + CUDA build work the g5g needed is the same platform those machines run."], []),
    (22, "AWS Silicon: Neuron", 0.5, [], []),
    (23, "Inferentia2: Gemma 4, Ported by Hand", 3.0,
     ["No vLLM release serves Gemma 4 on Neuron, so the model is compiled by hand.",
      "This one is running in us-west-2 if the rehearsal failover repeated."],
     [ask("inf2")]),
    (24, "Trainium: The Same Image, No Rebuild", 3.0,
     ["Same digest as inf2; the paragraph comes back word for word the same.",
      "36.9 against inf2's 36.6 tok/s."],
     [ask("trn1")]),
    (25, "vLLM on Neuron Lags the GPU Releases", 1.5,
     ["The newest vLLM-Neuron lists Trn2 and Trn3 only; Trn1/Inf2 stop at vLLM 0.16."], []),
    (26, "Why Neuron Is Hard to Serve On", 2.0,
     ["No vLLM release serves Gemma 4 on Neuron, so it runs a hand-ported engine with no tool calls.",
      "Everything compiles ahead of time for fixed shapes; a new shape costs ~3.9 s, so per-token shape changes are fatal.",
      "Memory is 2 x 16 GiB, one per NeuronCore; a gather above ~4-5 GB returns zeros without an error.",
      "int4 weights get a dense bf16 copy before each matmul, 27x per call. Net: 2-2.7x the L4's bandwidth, 0.28x its speed."], []),
    (27, "One Strands Agent", 0.5, [], []),
    (28, "One Agent, Four Model Classes", 2.0,
     ["BedrockModel, OpenAIModel with a base_url, SageMakerAIModel.",
      "Neuron backends are called as tools: no tool calls, 128-token prompt."], []),
    (29, "Demo 1: Every Backend Picked the Right Filter", 4.0,
     ["Eleven ids; how many are 10 or more? The answer is 8.",
      "count_ids does the counting. The grade checks the answer and the filter sent.",
      "Then the contrast: --tools rows hands the model the ids to count itself."],
     [{"cmd": "python3 demo1_counting.py --runs 3", "recorded": f"{RECORDED}/stage-demo1-engine.txt"},
      {"cmd": "python3 demo1_counting.py --runs 3 --tools rows", "recorded": f"{RECORDED}/stage-demo1-rows.txt"}]),
    (30, "Demo 2: The Table Was Right, the Sentence Was Wrong", 3.5,
     ["A Bedrock agent asks every backend; scoreboard() sorts in code.",
      "Read the table, then the agent's sentence. trn1 beat g6 in one of five rehearsal runs.",
      "The [harness] line lists any backend the agent skipped."],
     [{"cmd": "python3 demo2_orchestrator.py", "recorded": f"{RECORDED}/stage-demo2.txt"}]),
    (31, "Same Prompt: vLLM on g6 Decoded Fastest", 3.0,
     ["100 tokens, timed in code, network and prefill included.",
      "Three groups that hold from run to run: g6; Bedrock and SageMaker; g5g, trn1, inf2."],
     [{"cmd": "python3 bench_chat.py --repeats 3", "recorded": f"{RECORDED}/stage-bench.txt"}]),
    (32, "Per Token, the g6 Is the Cheapest Instance", 2.0,
     ["Arithmetic in cost.py: hourly list price over measured tokens per hour.",
      "g6: $1.69 per million one at a time, $0.150 at 16 in parallel; SageMaker about twice that.",
      "Bedrock Nova Micro is $0.14 per million output: a 24x7 g6 wins only above ~710 tok/s."], []),
    (33, "So, Which One?", 2.5,
     ["Bedrock first; g6 for your own weights; SageMaker to manage it; Neuron when the model exists for it."], []),
    (34, "Clean Up Four Regions", 1.5,
     ["stage/down.sh terminates, deletes the endpoint and sweeps four regions.",
      "Run it after the talk, not now."], []),
    (35, "Links", 1.0, ["Repo, rigs, write-up."], []),
    (36, "Questions", 5.0, [], []),
]


def configured() -> set:
    try:
        import backends
        return {b.name for b in backends.configured()}
    except Exception:
        return set()


def show_file(path: str) -> None:
    full = os.path.join(REPO, path)
    print(f"{YELLOW}[recorded run: {path}]{RESET}")
    try:
        with open(full) as f:
            sys.stdout.write(f.read())
    except OSError as exc:
        print(f"{RED}cannot read it: {exc}{RESET}")


def run(c: dict, have: set) -> bool:
    need = c.get("needs")
    if need and need not in have:
        print(f"{YELLOW}{need} is not configured (source stage/demo.env?). r shows the recorded run.{RESET}")
        return False
    print(f"{BOLD}$ {c['cmd']}{RESET}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(c["cmd"], shell=True, cwd=REPO)
    took = time.perf_counter() - t0
    if rc != 0:
        print(f"{RED}exit {rc} after {took:.1f} s. r shows the recorded run.{RESET}")
        return False
    print(f"{DIM}({took:.1f} s){RESET}")
    return True


def header(i: int, start: float) -> None:
    n, title, minutes, notes, cmds = CUES[i]
    planned = sum(c[2] for c in CUES[:i])
    elapsed = (time.time() - start) / 60
    drift = elapsed - planned
    pace = (f"{GREEN}on time{RESET}" if abs(drift) < 1 else
            f"{YELLOW}{drift:+.1f} min{RESET}" if abs(drift) < 3 else f"{RED}{drift:+.1f} min{RESET}")
    print("\n" + "─" * 72)
    print(f"{BOLD}Slide {n}/{len(CUES)}  {title}{RESET}   {DIM}{minutes:g} min{RESET}")
    print(f"{DIM}planned {planned:.1f} min · clock {elapsed:.1f} min · {RESET}{pace}")
    print(f"{DIM}{link(n)}{RESET}")
    for line in notes:
        print(f"  • {line}")
    for c in cmds:
        print(f"  {BOLD}▶ {c['cmd']}{RESET}")


def list_cues() -> None:
    total = sum(c[2] for c in CUES)
    for n, title, minutes, notes, cmds in CUES:
        print(f"{n:>2}  {minutes:>4g} min  {title}")
        for c in cmds:
            print(f"              $ {c['cmd']}")
    print(f"\n{len(CUES)} slides, {total:g} minutes planned, "
          f"{sum(len(c[4]) for c in CUES)} live commands")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="slide to start on")
    ap.add_argument("--list", action="store_true", help="print the cues and commands, run nothing")
    a = ap.parse_args()
    if a.list:
        list_cues()
        return 0

    have = configured()
    print(f"Backends configured: {', '.join(sorted(have)) or 'none (source stage/demo.env)'}")
    i = max(0, min(len(CUES) - 1, a.start - 1))
    # Starting mid-deck counts as being on schedule at that slide.
    start = time.time() - 60 * sum(c[2] for c in CUES[:i])
    step = 0
    header(i, start)
    while True:
        cmds = CUES[i][4]
        cur = cmds[step] if step < len(cmds) else None
        prompt = f"[Enter: run {cur['cmd'].removeprefix('python3 ')}]" if cur else "[Enter: next slide]"
        try:
            key = input(f"{DIM}{prompt} n p g N r s l q > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if key == "q":
            return 0
        if key == "" and cur:
            run(cur, have)
            step += 1
            continue
        if key == "r" and cur:
            show_file(cur["recorded"])
            step += 1
            continue
        if key == "r" and step > 0 and cmds:
            show_file(cmds[step - 1]["recorded"])
            continue
        if key == "s" and cur:
            step += 1
            continue
        if key == "l":
            print(link(CUES[i][0]))
            continue
        if key in ("", "n"):
            if i == len(CUES) - 1:
                print("Last slide. q to quit.")
                continue
            i, step = i + 1, 0
        elif key == "p":
            i, step = max(0, i - 1), 0
        elif key.startswith("g"):
            try:
                i, step = max(0, min(len(CUES) - 1, int(key[1:].strip()) - 1)), 0
            except ValueError:
                print("g needs a slide number, e.g. g 23")
                continue
        else:
            print("Keys: Enter n p g N r s l q")
            continue
        header(i, start)


if __name__ == "__main__":
    sys.exit(main())
