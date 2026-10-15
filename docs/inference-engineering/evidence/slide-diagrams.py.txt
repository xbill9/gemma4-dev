"""Architecture diagrams for the deck's section slides, drawn in code so every label is exact.

    python3 docs/slides/make_diagrams.py     writes docs/slides/arch-*.png (3200 x 880 px); the deck uses copies named with their hash

Each diagram is one or two rows of boxes joined by arrows: what runs on the laptop, how the
request travels, and what serves it. Labels come from backends.py, stage/up.sh and the 2026-10-12
rehearsal (results/rehearsal-20261012T1429Z).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, NAVY, ORANGE, TEAL, BG = "#1E1E1E", "#232F3E", "#FF9900", "#1A9C8C", "#EEEEEE"
FILL = {"laptop": "#FFFFFF", "aws": "#FFF1DB", "path": "#E3F4F1", "chip": "#232F3E"}
W, H = 32, 8.8  # data units; a 16 x 4.4 in figure at 200 dpi -> 3200 x 880 px
FIG, DPI = (16, 4.4), 200
plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans"]


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=FILL[kind], ec=NAVY if kind != "path" else TEAL, lw=3))
    color = "white" if kind == "chip" else INK
    ax.text(x + w / 2, y + h - 0.95, title, ha="center", va="center", fontsize=21, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 1.55, sub, ha="center", va="top", fontsize=16, color=color, linespacing=1.25)


def arrow(ax, x0, x1, y, label=""):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=34, lw=3.5, color=ORANGE))
    if label:
        ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="bottom", fontsize=15, color=INK)


def row(ax, y, h, items):
    """items: [(title, sub, kind, width, arrow_label_to_next)] laid out left to right."""
    gap = 2.4 if len(items) > 3 else 3.2
    total = sum(i[3] for i in items)
    x = (W - total - gap * (len(items) - 1)) / 2
    for i, (title, sub, kind, w, label) in enumerate(items):
        box(ax, x, y, w, h, title, sub, kind)
        if i < len(items) - 1:
            arrow(ax, x + w + 0.1, x + w + gap - 0.1, y + h / 2, label)
        x += w + gap


def canvas():
    fig = plt.figure(figsize=FIG, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    return fig, ax


def figure(name, rows):
    fig, ax = canvas()
    for y, h, items in rows:
        row(ax, y, h, items)
    path = os.path.join(OUT, f"arch-{name}.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)


def main():
    # Slide 11: the two managed paths
    figure("managed", [
        (4.8, 3.2, [("Strands agent", "BedrockModel", "laptop", 5.6, "Converse API"),
                    ("Amazon Bedrock", "us-east-1, per token", "aws", 7.4, ""),
                    ("Nova Micro", "managed model", "chip", 5.6, "")]),
        (0.8, 3.2, [("Strands agent", "SageMakerAIModel", "laptop", 5.6, "InvokeEndpoint"),
                    ("SageMaker endpoint", "ml.g6.xlarge, us-east-2\nAWS vLLM 0.31 container", "aws", 7.4, ""),
                    ("NVIDIA L4", "Gemma 4 E2B repack", "chip", 5.6, "")]),
    ])
    # Slide 14: vLLM on EC2 GPUs, reached through SSM
    figure("gpus", [
        (4.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("g6.xlarge", "x86_64 host\nvLLM 0.30", "aws", 6.4, ""),
                    ("NVIDIA L4", "24 GB\nE2B repack", "chip", 5.4, "")]),
        (0.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("g5g.2xlarge", "Graviton2 Arm host\nvLLM 0.27.2rc0, sm_75", "aws", 6.4, ""),
                    ("NVIDIA T4G", "16 GB\nE2B fp16", "chip", 5.4, "")]),
    ])
    # Slide 17: the hand-ported Neuron server
    figure("neuron", [
        (2.7, 3.4, [("Strands agent", "plain HTTP tool\n(demo 2 only)", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("inf2 or trn1", "gemma4-optb:slim\nhand-ported Gemma 4", "aws", 6.4, ""),
                    ("2 NeuronCores", "NeuronCore-v2\nsame image, both chips", "chip", 5.4, "")]),
    ])
    # Slide 21: one agent, six backends, one row each
    fig, ax = canvas()
    box(ax, 0.8, 2.9, 7.0, 3.0, "One Strands agent", "a model object\nper backend", "laptop")
    backends = [("Bedrock", "BedrockModel", "aws"), ("SageMaker", "SageMakerAIModel", "aws"),
                ("EC2 g6, NVIDIA L4", "OpenAIModel", "aws"), ("EC2 g5g, NVIDIA T4G", "OpenAIModel", "aws"),
                ("Inferentia2", "plain HTTP tool", "chip"), ("Trainium", "plain HTTP tool", "chip")]
    h, step = 1.18, 1.4
    for i, (name, cls, kind) in enumerate(backends):
        y = H - 0.4 - h - i * step
        ax.add_patch(FancyBboxPatch((14.5, y), 16.5, h, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=FILL[kind], ec=NAVY, lw=3))
        color = "white" if kind == "chip" else INK
        ax.text(15.1, y + h / 2, name, ha="left", va="center", fontsize=24, weight="bold", color=color)
        ax.text(30.4, y + h / 2, cls, ha="right", va="center", fontsize=19, color=color)
        ax.add_patch(FancyArrowPatch((7.9, 4.4), (14.4, y + h / 2), arrowstyle="-|>", mutation_scale=28,
                                     lw=3, color=ORANGE))
    path = os.path.join(OUT, "arch-overview.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)


def ways():
    """Slide 4: one agent, four ways to run a model, from managed to your own chip."""
    w, h = 32, 13.2
    fig = plt.figure(figsize=(16, 6.6), dpi=DPI, facecolor="white")  # sits on a white card
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(0, h), ax.axis("off")
    ax.add_patch(FancyBboxPatch((10, 10.7), 12, 2.1, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=FILL["laptop"], ec=NAVY, lw=3))
    ax.text(16, 12.05, "One Strands agent", ha="center", va="center", fontsize=26, weight="bold", color=INK)
    ax.text(16, 11.25, "same question, graded in code", ha="center", va="center", fontsize=18, color=INK)
    cols = [("Managed API", ["Amazon Bedrock", "Nova Micro, per token"]),
            ("Managed endpoint", ["SageMaker", "ml.g6.xlarge, vLLM"]),
            ("Your GPU", ["EC2 g6: NVIDIA L4", "EC2 g5g: Arm + T4G"]),
            ("Your AWS chip", ["Inferentia2", "Trainium"])]
    cw, gap, top, bot = 6.8, 1.0, 8.6, 3.4
    for i, (head, lines) in enumerate(cols):
        x = 1.2 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, bot), cw, top - bot, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=FILL["aws"], ec=NAVY, lw=3))
        ax.add_patch(FancyBboxPatch((x, top - 1.6), cw, 1.6, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=NAVY, ec=NAVY, lw=3))
        ax.text(x + cw / 2, top - 0.8, head, ha="center", va="center", fontsize=22, weight="bold", color="white")
        ax.text(x + cw / 2, top - 2.75, lines[0], ha="center", va="center", fontsize=20, weight="bold", color=INK)
        ax.text(x + cw / 2, top - 4.0, lines[1], ha="center", va="center", fontsize=18,
                weight="bold" if i >= 2 else "normal", color=INK)
        ax.add_patch(FancyArrowPatch((16, 10.6), (x + cw / 2, top + 0.1), arrowstyle="-|>", mutation_scale=28,
                                     lw=3, color=ORANGE))
    ax.add_patch(FancyArrowPatch((1.2, 2.2), (30.8, 2.2), arrowstyle="-|>", mutation_scale=40, lw=5, color=TEAL))
    ax.text(1.2, 1.0, "nothing to run, pay per token", ha="left", va="center", fontsize=19, color=INK)
    ax.text(30.8, 1.0, "more control, more to run", ha="right", va="center", fontsize=19, color=INK)
    path = os.path.join(OUT, "arch-ways.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(path)


def precision():
    """Slide 6: what fewer bits buy (memory), what they cost (quality), and where they run natively.

    Memory: E2B weights in GPU memory per build, gemma4-dev gpu-vllm-mi300x-2b dtype sweep 2026-10-07
    ("Model loading took", GiB). Quality: share of the 3,880-record suite read correctly on one TPU v5e,
    gemma4-dev tpu-vllm-v5e1-2b-* READMEs, 2026-09-29. Native math: L4 and T4G from HARDWARE.md and
    NVIDIA; NeuronCore-v2 from the AWS Neuron data-types and architecture pages.
    """
    fig = plt.figure(figsize=(16, 6.75), dpi=DPI, facecolor="white")
    mem = [("bf16", 9.42), ("fp8", 7.07), ("int8", 7.07), ("int4", 6.32), ("int4 + tables", 2.85)]
    qual = [("int8, from QAT weights", 0.686), ("bf16", 0.683), ("int4 + tables", 0.681),
            ("int4, QAT", 0.678), ("fp8, from QAT weights", 0.670), ("int8, rounded from bf16", 0.670)]
    chips = ["L4", "T4G", "Neuron"]
    fmts = ["bf16", "fp8", "int8", "int4"]
    grid = {"L4": ["yes", "yes", "yes", "widened"], "T4G": ["emulated", "no", "yes", "widened"],
            "Neuron": ["yes", "pending", "yes", "widened"]}
    color = {"yes": TEAL, "no": "#C0392B", "emulated": ORANGE, "widened": ORANGE, "pending": ORANGE}

    ax = fig.add_axes([0.11, 0.30, 0.20, 0.56])
    names, vals = [m[0] for m in mem][::-1], [m[1] for m in mem][::-1]
    ax.barh(names, vals, color=[TEAL if v < 3 else NAVY for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:g} GiB", va="center", fontsize=15, color=INK)
    ax.set_xlim(0, 14), ax.tick_params(labelsize=15, length=0), ax.set_xticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("Memory: E2B weights", fontsize=20, weight="bold", color=INK, loc="left")

    ax = fig.add_axes([0.50, 0.30, 0.17, 0.56])
    names, vals = [q[0] for q in qual][::-1], [q[1] for q in qual][::-1]
    ax.hlines(range(len(vals)), 0.66, vals, color="#DDDDDD", lw=3)
    ax.scatter(vals, range(len(vals)), s=180, color=[TEAL if v >= 0.678 else ORANGE for v in vals], zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.0018, i, f"{v:.3f}", va="center", fontsize=15, color=INK)
    ax.set_yticks(range(len(names)), names), ax.set_xlim(0.66, 0.695), ax.set_xticks([])
    ax.tick_params(labelsize=14, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("Quality: suite accuracy", fontsize=20, weight="bold", color=INK, loc="left")

    ax = fig.add_axes([0.77, 0.30, 0.21, 0.56])
    ax.set_xlim(0, 4), ax.set_ylim(0, 3.6), ax.axis("off")
    for j, f in enumerate(fmts):
        ax.text(j + 0.5, 3.3, f, ha="center", va="center", fontsize=16, weight="bold", color=INK)
    for i, c in enumerate(chips):
        y = 2.5 - i
        ax.text(-0.1, y, c, ha="right", va="center", fontsize=16, weight="bold", color=INK)
        for j, v in enumerate(grid[c]):
            ax.add_patch(FancyBboxPatch((j + 0.06, y - 0.38), 0.88, 0.76, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=color[v], ec="none", alpha=0.9))
            ax.text(j + 0.5, y, v, ha="center", va="center", fontsize=11 if len(v) > 7 else 12.5, weight="bold", color="white")
    ax.set_title("Native math on the chip", fontsize=20, weight="bold", color=INK, loc="left")

    for x, t in [(0.02, "Fewer bits: less memory,\ndisk and bandwidth"),
                 (0.36, "int4 trained for it (QAT): within\nhalf a point of bf16"),
                 (0.70, "Faster only where the chip\nmultiplies that format")]:
        fig.text(x, 0.06, t, fontsize=17, color=INK, weight="bold", linespacing=1.3)
    path = os.path.join(OUT, "precision.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(path)


def sagemaker():
    """SageMaker intro slide: the three objects you create and what the endpoint runs for you.

    From sagemaker-gemma/sm.py deploy(): create-model (image + SM_VLLM_* env + execution role),
    create-endpoint-config (one variant, ml.g6.xlarge x 1), create-endpoint. The vLLM container
    downloads the weights named by SM_VLLM_MODEL from Hugging Face at start-up.
    """
    w, h = 32, 13.5
    fig = plt.figure(figsize=(16, 6.75), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(0, h), ax.axis("off")

    def rbox(x, y, bw, bh, fc, ec, lw=3, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=fc, ec=ec, lw=lw, ls=ls))

    def label(x, y, t, size=20, bold=True, color=INK, ha="center"):
        ax.text(x, y, t, ha=ha, va="center", fontsize=size, weight="bold" if bold else "normal", color=color)

    # left: what you create
    label(4.6, 12.6, "You create", 22)
    for i, (t, sub) in enumerate([("Model", "vLLM container image\nSM_VLLM_* settings, IAM role"),
                                  ("Endpoint config", "ml.g6.xlarge x 1"),
                                  ("Endpoint", "a name you call")]):
        y = 9.0 - i * 3.3
        rbox(0.8, y, 7.6, 2.7, FILL["aws"], NAVY)
        label(4.6, y + 1.9, t, 21)
        label(4.6, y + 0.85, sub, 15, bold=False)
    ax.add_patch(FancyArrowPatch((8.6, 6.4), (11.2, 6.4), arrowstyle="-|>", mutation_scale=34, lw=3.5, color=ORANGE))

    # middle: what SageMaker runs
    rbox(11.4, 1.0, 12.4, 11.0, "#F7F9FA", ORANGE, lw=3, ls=(0, (6, 4)))
    label(17.6, 12.6, "SageMaker runs", 22)
    rbox(12.4, 7.4, 10.4, 3.6, FILL["laptop"], NAVY)
    label(17.6, 10.2, "ml.g6.xlarge instance", 20)
    rbox(13.0, 7.9, 4.6, 1.6, FILL["path"], TEAL)
    label(15.3, 8.7, "vLLM container", 15)
    rbox(18.2, 7.9, 4.0, 1.6, FILL["chip"], NAVY)
    label(20.2, 8.7, "NVIDIA L4", 15, color="white")
    for i, t in enumerate(["health check before traffic", "logs and metrics in CloudWatch", "instance count you can scale"]):
        label(12.8, 5.9 - i * 1.5, "\u2022  " + t, 16, bold=False, ha="left")

    # right: client and weights
    rbox(25.2, 8.6, 6.2, 2.8, FILL["laptop"], NAVY)
    label(28.3, 10.4, "Strands agent", 19)
    label(28.3, 9.4, "SageMakerAIModel", 15, bold=False)
    ax.add_patch(FancyArrowPatch((25.1, 9.6), (22.9, 9.6), arrowstyle="-|>", mutation_scale=30, lw=3.5, color=ORANGE))
    label(28.3, 7.6, "InvokeEndpoint, signed HTTPS", 14, bold=False)
    rbox(25.2, 2.2, 6.2, 2.8, FILL["laptop"], NAVY)
    label(28.3, 4.0, "Hugging Face", 19)
    label(28.3, 3.0, "weights at start-up", 15, bold=False)
    ax.add_patch(FancyArrowPatch((25.1, 3.6), (17.6, 7.3), arrowstyle="-|>", mutation_scale=30, lw=3, color=TEAL,
                                 connectionstyle="arc3,rad=0.15"))

    path = os.path.join(OUT, "sagemaker.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(path)

if __name__ == "__main__":
    main()
    ways()
    precision()
    sagemaker()
