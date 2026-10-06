import sys, os, types
print("IMPORT pid=%d ppid=%d name=%s child=%s" % (os.getpid(), os.getppid(), __import__("multiprocessing").current_process().name, os.environ.get("_TP_CHILD")), flush=True)
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing, torch
import neuronx_distributed
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
H = 512
class Block(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.up=ColumnParallelLinear(H,4*H,bias=False,gather_output=False); s.down=RowParallelLinear(4*H,H,bias=False,input_is_parallel=True)
    def forward(s,x): return s.down(torch.nn.functional.gelu(s.up(x)))
def get_callable(): return Block().eval(), {}
def main():
    print("MAIN_ENTER pid=%d" % os.getpid(), flush=True)
    ex=(torch.randn(1,8,H),)
    traced=neuronx_distributed.trace.parallel_model_trace(get_callable, ex, tp_degree=2, compiler_args=["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"])
    out=traced(*ex); print("OUTPUT SHAPE", tuple(out.shape), flush=True); print("TP_PROBE_OK", flush=True)
if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
    else:
        print("SKIP_MAIN child pid=%d" % os.getpid(), flush=True)
