#!/usr/bin/env python3
"""Clamp Triton's unified-attention tiles so they fit Turing's 64 KiB per block.

THIS IS THE ONE THING THIS RIG HAS TO DO THAT ITS SIBLINGS DO NOT, and it is the
whole reason the rig exists. Read `docs/turing-shared-memory.md` before changing
anything here.

The problem, MEASURED on `gpu-vllm-g5g-2b` 2026-08-12 on the same Turing silicon:

    Gemma4 model has heterogeneous head dimensions
    {'sliding_attention': 256, 'full_attention': 512}. FA4 not available,
    forcing TRITON_ATTN.
    triton.runtime.errors.OutOfResources: shared memory,
    Required: 98304, Hardware limit: 65536

Gemma 4's global-attention layers are 512 wide, only FA4 or Triton handle
heterogeneous head dims, FA4 is unavailable, so vLLM FORCES Triton and the choice
is not overridable. Triton's tile at head_size=512 wants ~96 KiB per block and
Turing allows at most 64 KiB. There is no flag; the tiles have to come down.

VERIFIED AGAINST REAL UPSTREAM SOURCE 2026-08-29 (v0.28.0 and main, identical):
`_get_tile_size` has NO shared-memory awareness -- it returns 32/16/32 from
head_size and element size alone -- so upstream has not fixed this and the clamp
is still required.

WHY A SCRIPT AND NOT A BUILD. The G5g sibling gets this patch in by compiling
vLLM from source, ~67 minutes, because the published arm64 image has no SM 7.5
kernels at all. On g4dn the published amd64 image ALREADY carries 7.5, so nothing
needs compiling -- only this one pure-Python file needs editing, and a derived
image built `FROM` the stock one takes seconds. That asymmetry is the point of
this rig.

DESIGN RULE: FAIL LOUDLY, NEVER SILENTLY NO-OP. A patch that quietly matches
nothing -- or that lands in the wrong PLACE -- leaves an unpatched engine behind
a patched-looking tag, and the failure then surfaces ~10 minutes later as an
OutOfResources at engine start. Every anchor and identifier is checked, and a
miss is exit code 2 with the surrounding source attached.

WHERE THE CLAMP GOES IS AS IMPORTANT AS WHAT IT SAYS, and this is the subtle one.
Upstream reads the tile constants into a local well before the kernel launch:

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (...); tile_size = TILE_SIZE_PREFILL     # <- consumed here
    else:
        grid = (...); tile_size = TILE_SIZE_DECODE      # <- and here
    launch_kwargs = {}
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages # <- and here
    kernel_unified_attention[grid](...)                 # <- the launch

**Inserting at the launch site would clamp three variables nothing reads
afterwards.** The marker would be present, the in-image verification would pass,
`verify_triton_patch` would report success, and the kernel would still ask for
98,304 bytes. So the insertion point is derived from the code rather than from a
launch-site pattern: immediately after the LAST assignment to either tile
constant, with an explicit check that nothing READS them before that point.

Usage:
    patch_triton_turing.py PATH            # patch in place (idempotent)
    patch_triton_turing.py --check PATH    # 0 = patched, 1 = not patched
"""

import ast
import os
import re
import sys

# The rig name is in the marker on purpose: an operator grepping a container for
# "why is this file different from upstream" gets an answer that names the rig.
MARKER = "# gpu-vllm-g4dn-2b: Turing shared-memory clamp"

# Headroom under the hard 65,536. The kernel's accumulators and Triton's pipeline
# buffers are not counted by the tile arithmetic below, so budgeting the full
# limit still overflows. 60000 is the value the G5g sibling ran with.
SMEM_BUDGET = int(os.getenv("TURING_SMEM_BUDGET", "60000"))

# The two tile constants the clamp rewrites. Insertion is anchored on their last
# assignment, so these names are load-bearing twice over.
TILES = ("TILE_SIZE_PREFILL", "TILE_SIZE_DECODE")

# Other identifiers the inserted block reads. If any is absent the file has been
# restructured upstream and the clamp would be nonsense -- refuse rather than
# insert code that raises NameError at the first token.
REQUIRED = ("current_platform", "BLOCK_M", "head_size")

# Triton multiplies shared memory by the pipeline depth, so the stage count is
# part of the budget rather than a nicety. The variable holding it has been
# renamed upstream before, so it is detected rather than assumed; override with
# TRITON_STAGES_VAR when the detection is ambiguous.
#
# Detected through `ast`, NOT a regex, and the difference is load-bearing: the
# launch site reads `launch_kwargs["num_stages"] = launch_num_stages`, which a
# line-oriented pattern reads as an assignment to `num_stages`. It is a subscript
# store. Picking it would write `num_stages = 1` into the enclosing scope, where
# it binds a local nothing reads, and the clamp would then be HALF applied while
# reporting success -- the precise failure mode this whole script exists to refuse.
STAGES_NAME = re.compile(r"\w*num_stages")


def fail(message: str, context: str = "") -> int:
    print(f"patch_triton_turing: ERROR: {message}", file=sys.stderr)
    if context:
        print("--- context ---", file=sys.stderr)
        print(context, file=sys.stderr)
    print(
        "REFUSING TO WRITE. An unpatched engine behind a patched tag fails ~10 minutes "
        "later as `OutOfResources: shared memory` at engine start.",
        file=sys.stderr,
    )
    return 2


def assigned_names(tree: ast.AST) -> "set[str]":
    """Names that are the TARGET of a real assignment statement.

    Subscript stores and keyword arguments are excluded -- see STAGES_NAME.
    """
    names = set()
    for node in ast.walk(tree):
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def resolve_stages_var(tree: ast.AST, text: str) -> "tuple[str, str]":
    """Return (variable_name, note). Empty name means 'could not decide'."""
    override = os.getenv("TRITON_STAGES_VAR", "").strip()
    if override:
        if override not in text:
            return "", f"TRITON_STAGES_VAR={override!r} does not appear in the file"
        return override, f"from TRITON_STAGES_VAR={override!r}"
    candidates = sorted(n for n in assigned_names(tree) if STAGES_NAME.fullmatch(n))
    if len(candidates) == 1:
        return candidates[0], "auto-detected as an assignment target"
    return "", f"expected exactly one assigned *num_stages name, found {candidates or 'none'}"


def find_target_function(tree: ast.AST) -> "ast.FunctionDef | None":
    """The function that assigns BOTH tile constants — that is the launcher."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and set(TILES) <= assigned_names(node):
            # Prefer the outermost/largest match if several qualify.
            if best is None or (node.end_lineno - node.lineno) > (best.end_lineno - best.lineno):
                best = node
    return best


def tile_assignment_lines(func: ast.FunctionDef) -> "list[int]":
    lines = []
    for node in ast.walk(func):
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in TILES:
                lines.append(node.lineno)
    return sorted(lines)


def tile_read_lines(func: ast.FunctionDef) -> "list[int]":
    """Lines where a tile constant is LOADED rather than stored."""
    return sorted(
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Name) and node.id in TILES and isinstance(node.ctx, ast.Load)
    )


def find_insertion_point(func: ast.FunctionDef) -> "tuple[int, int, str]":
    """Return (1-based insert-before line, indent columns, note).

    Immediately after the last function-body statement that assigns a tile
    constant anywhere inside it, so the clamp sees final values and every later
    read sees clamped ones.
    """
    last_index = None
    for index, stmt in enumerate(func.body):
        names = assigned_names(stmt)
        if names & set(TILES):
            last_index = index
    if last_index is None:
        return 0, 0, "no statement in the function body assigns a tile constant"
    anchor = func.body[last_index]
    indent = anchor.col_offset
    return anchor.end_lineno + 1, indent, (
        f"after the statement ending at line {anchor.end_lineno} "
        f"({type(anchor).__name__})"
    )


def build_clamp(indent: str, stages_var: str) -> str:
    body = [
        f"{indent}{MARKER}",
        f"{indent}# Turing (SM 7.5) caps a block at 65,536 B of shared memory, and only",
        f"{indent}# if the kernel opts into the dynamic attribute -- the static default is",
        f"{indent}# 49,152. Triton wants 98,304 at head_size=512, which is Gemma 4's",
        f"{indent}# global-attention width. Halve the tiles until they fit and drop the",
        f"{indent}# pipeline to one stage. Pre-Ampere only: a no-op on every other device.",
        f"{indent}#",
        f"{indent}# Placed here, after the LAST tile assignment and before the first read,",
        f"{indent}# on purpose. Clamping at the launch site would rewrite variables that",
        f"{indent}# have already been copied into `tile_size` and `launch_kwargs`.",
        f"{indent}if current_platform.get_device_capability()[0] < 8:",
        f"{indent}    _smem_budget = {SMEM_BUDGET}",
        f"{indent}    _esz = q.element_size()",
        "",
        f"{indent}    def _fits(tile):",
        f"{indent}        return (BLOCK_M + 2 * tile) * head_size * _esz <= _smem_budget",
        "",
        f"{indent}    while TILE_SIZE_PREFILL > 16 and not _fits(TILE_SIZE_PREFILL):",
        f"{indent}        TILE_SIZE_PREFILL //= 2",
        f"{indent}    while TILE_SIZE_DECODE > 16 and not _fits(TILE_SIZE_DECODE):",
        f"{indent}        TILE_SIZE_DECODE //= 2",
        f"{indent}    {stages_var} = 1",
        "",
    ]
    return "\n".join(body)


def main(argv: "list[str]") -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    check_only = "--check" in argv[1:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = args[0]

    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return fail(f"cannot read {path}: {exc}")

    if MARKER in text:
        print(f"patch_triton_turing: already patched: {path}")
        return 0
    if check_only:
        print(f"patch_triton_turing: NOT patched: {path}", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED if name not in text]
    if missing:
        return fail(
            f"{path} does not mention {', '.join(missing)} — upstream has been "
            "restructured and this patch no longer describes it"
        )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return fail(f"{path} does not parse as Python: {exc}")

    func = find_target_function(tree)
    if func is None:
        return fail(
            f"no function in {path} assigns both {' and '.join(TILES)} — the tile "
            "constants have been renamed or moved out of the launcher"
        )

    insert_line, indent_cols, note = find_insertion_point(func)
    if not insert_line:
        return fail(note)

    # THE CHECK THAT MAKES THE PLACEMENT VERIFIED RATHER THAN HOPED FOR.
    # Every read of a tile constant must come AFTER the clamp, or the clamp is
    # rewriting values something has already consumed.
    early_reads = [ln for ln in tile_read_lines(func) if ln >= insert_line]
    premature = [
        ln
        for ln in tile_read_lines(func)
        if ln < insert_line and ln not in set(tile_assignment_lines(func))
    ]
    if premature:
        context = "\n".join(
            f"{n:5d}: {line}"
            for n, line in enumerate(text.splitlines(), 1)
            if n in set(premature)
        )
        return fail(
            f"tile constants are READ at line(s) {premature} before the insertion "
            f"point at line {insert_line}; clamping there would be too late",
            context,
        )
    if not early_reads:
        return fail(
            f"nothing reads {' or '.join(TILES)} after line {insert_line} — the clamp "
            "would have no effect, which is the silent half-fix this script refuses"
        )

    stages_var, stages_note = resolve_stages_var(tree, text)
    if not stages_var:
        return fail(
            f"cannot identify the pipeline-stage variable ({stages_note}). Triton counts "
            "shared memory per stage, so clamping tiles alone may still overflow. "
            "Set TRITON_STAGES_VAR to the right name and re-run."
        )

    lines = text.splitlines(keepends=True)
    lines.insert(insert_line - 1, build_clamp(" " * indent_cols, stages_var))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))

    print(
        f"patch_triton_turing: patched {path}\n"
        f"  function      : {func.name} (lines {func.lineno}-{func.end_lineno})\n"
        f"  inserted before line {insert_line}, {note}\n"
        f"  indent        : {indent_cols} spaces\n"
        f"  stages var    : {stages_var} ({stages_note})\n"
        f"  tile reads after the clamp: {early_reads}\n"
        f"  smem budget   : {SMEM_BUDGET} B of Turing's 65536 hard limit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
