"""TP=2 sharded Gemma4-E2B prefill validation.
Shards q/o/gate/up/down across 2 cores (k/v replicated, MQA nkv=1), traces the lang
forward via NxD parallel_model_trace, compares first-token argmax to a CPU fp32 reference."""
import sys, os, types
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing

MP = "/workspace/real-gemma4-E2B-it"
TP = 2
PROMPT = "What is the capital of France?"

def _gelu_patch(lang):
    import torch
    class GeluTanh(torch.nn.Module):
        def forward(s, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()

def _load_lang(dtype):
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=dtype, attn_implementation="eager"); mm.eval()
    _gelu_patch(mm.model.language_model)
    return mm

def get_callable():
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    mm = _load_lang(torch.float32)
    lang = mm.model.language_model; lm_head = mm.lm_head; cfg = lang.config
    softcap = getattr(mm.config.text_config, "final_logit_softcapping", None)

    def col(orig):
        out, inf = orig.weight.shape
        l = ColumnParallelLinear(inf, out, bias=False, gather_output=False)
        c = out // TP
        l.weight.data.copy_(orig.weight.data[rank*c:(rank+1)*c, :])
        return l
    def row(orig):
        out, inf = orig.weight.shape
        l = RowParallelLinear(inf, out, bias=False, input_is_parallel=True)
        c = inf // TP
        l.weight.data.copy_(orig.weight.data[:, rank*c:(rank+1)*c])
        return l

    for lyr in lang.layers[:cfg.num_hidden_layers]:
        a = lyr.self_attn
        a.q_proj = col(a.q_proj)
        a.o_proj = row(a.o_proj)
        a.num_key_value_groups = a.num_key_value_groups // TP  # 8 -> 4 (q heads/rank vs replicated kv)
        mlp = lyr.mlp
        mlp.gate_proj = col(mlp.gate_proj)
        mlp.up_proj = col(mlp.up_proj)
        mlp.down_proj = row(mlp.down_proj)

    def softcap_logits(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg

    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.lang = lang; s.head = lm_head
        def forward(s, ie, am, ple):
            from transformers import DynamicCache
            out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am,
                         use_cache=True, past_key_values=DynamicCache())
            return softcap_logits(s.head(out.last_hidden_state))
    return Wrap().eval(), {}

def main():
    import torch
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    import neuronx_distributed
    tok = AutoTokenizer.from_pretrained(MP)
    ref = _load_lang(torch.float32)
    rlang = ref.model.language_model; rhead = ref.lm_head
    softcap = getattr(ref.config.text_config, "final_logit_softcapping", None)
    enc = tok.apply_chat_template([{"role":"user","content":PROMPT}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"]; n0 = ids.shape[1]
    with torch.no_grad():
        ie = rlang.embed_tokens(ids); ple = rlang.get_per_layer_inputs(ids, ie)
    am = torch.ones(1, n0, dtype=torch.long)
    # CPU reference first token
    with torch.no_grad():
        from transformers import DynamicCache
        ro = rlang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am, use_cache=True, past_key_values=DynamicCache())
        rlg = softcap*torch.tanh(rhead(ro.last_hidden_state)/softcap) if softcap else rhead(ro.last_hidden_state)
    ref_first = int(rlg[0, n0-1].argmax())
    print("prompt tokens", n0, "| CPU ref first token", ref_first, repr(tok.decode([ref_first])), flush=True)

    print("tracing TP=2 prefill ...", flush=True)
    traced = neuronx_distributed.trace.parallel_model_trace(
        get_callable, (ie, am, ple), tp_degree=TP,
        compiler_args=["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"])
    dlg = traced(ie, am, ple)
    dev_first = int(dlg[0, n0-1].argmax())
    print("DEVICE TP first token", dev_first, repr(tok.decode([dev_first])), flush=True)
    print("PREFILL_MATCH", dev_first == ref_first, flush=True)
    print("TP_MODEL_OK", flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"] = "1"
        multiprocessing.set_start_method("spawn", force=True)
        main()
