"""Reconstruct Gemma-4-E2B KV-share machinery on a STOCK NxDI 0.10.0 install.

Run INSIDE the neuron container (public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0).
Stock NxDI 0.10.0 gemma3 has PLE + hybrid head_dim + Gemma scaling + QK-norm natively,
but LACKS layer-level KV-sharing (num_kv_shared_layers). This adds it (3 edits).
Anchors: sliding=13, full=14; first_shared=15 (num_hidden_layers 35 - num_kv_shared_layers 20).
Confirmed vs vLLM gemma3n: offset = 2 if sliding else 1 -> targets 13 (sliding) / 14 (full).
Backs up each file to *.stock.bak. Idempotent-ish (counts must be 1).
"""
import shutil, os
D="/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference"
G=D+"/models/gemma3/modeling_gemma3.py"; A=D+"/modules/attention/attention_base.py"; MB=D+"/models/model_base.py"
for f in (G,A,MB):
    if not os.path.exists(f+".stock.bak"): shutil.copy(f,f+".stock.bak")
res={}
# A) decoder layer: pass layer_idx to attention + set gemma4 KV-share flags on it
g=open(G).read()
oA="        self.self_attn = NeuronGemma3Attention(config)"
nA=('        self.self_attn = NeuronGemma3Attention(config)\n'
    '        _a = self.self_attn\n'
    '        _a.gemma4_layer_type = "sliding" if (config.sliding_window is not None and (layer_idx is None or (layer_idx + 1) % 5 != 0)) else "full"\n'
    '        _nkvs = getattr(config, "num_kv_shared_layers", 0) or 0\n'
    '        _first = config.num_hidden_layers - _nkvs\n'
    '        _a.gemma4_shared_kv = (layer_idx is not None) and (_nkvs > 0) and (layer_idx >= _first)\n'
    '        _asl = max([j for j in range(_first) if (j + 1) % 5 != 0], default=-1)\n'
    '        _afl = max([j for j in range(_first) if (j + 1) % 5 == 0], default=-1)\n'
    '        _a.gemma4_store_kv = (_nkvs > 0) and (layer_idx in (_asl, _afl))\n'
    '        _a.gemma4_v_norm = (not _a.gemma4_shared_kv)\n'
    '        _a.gemma4_rms_eps = config.rms_norm_eps')
res['A_flags']=g.count(oA); g=g.replace(oA,nA,1); open(G,"w").write(g)
# B) attention_base prep_qkv_tensors: v_norm on V + eager KV-share stash (globals) before return
a=open(A).read()
oB="        return Q, K, V, cos_cache, sin_cache, residual"
nB=('        if getattr(self, "gemma4_v_norm", False):\n'
    '            _vf = V.to(torch.float32); _vms = _vf.pow(2).mean(-1, keepdim=True) + getattr(self, "gemma4_rms_eps", 1e-6); V = (_vf * torch.pow(_vms, -0.5)).to(V.dtype)\n'
    '        _g4s = globals().setdefault("_GEMMA4_KV_STASH", {})\n'
    '        if getattr(self, "gemma4_store_kv", False):\n'
    '            _g4s[self.gemma4_layer_type] = (K, V)\n'
    '        elif getattr(self, "gemma4_shared_kv", False):\n'
    '            K, V = _g4s[self.gemma4_layer_type]\n'
    '        return Q, K, V, cos_cache, sin_cache, residual')
res['B_stash']=a.count(oB); a=a.replace(oB,nB,1); open(A,"w").write(a)
# C) model_base loop: _gemma4_src_idx cache redirect (token-gen/history sharing)
m=open(MB).read()
oC="            past_key_value = past_key_values[idx] if past_key_values is not None else None"
nC=('            _gsrc = idx\n'
    '            _ga = getattr(decoder_layer, "self_attn", None)\n'
    '            if _ga is not None and getattr(_ga, "gemma4_shared_kv", False):\n'
    '                _gt = getattr(_ga, "gemma4_layer_type", None)\n'
    '                for _j, _dl in enumerate(self.layers):\n'
    '                    _da = getattr(_dl, "self_attn", None)\n'
    '                    if _da is not None and getattr(_da, "gemma4_store_kv", False) and getattr(_da, "gemma4_layer_type", None) == _gt:\n'
    '                        _gsrc = _j\n'
    '            past_key_value = past_key_values[_gsrc] if past_key_values is not None else None')
res['C_redirect']=m.count(oC); m=m.replace(oC,nC,1); open(MB,"w").write(m)
print(res)  # expect all 1
