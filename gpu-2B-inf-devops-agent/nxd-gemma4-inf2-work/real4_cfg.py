
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import Gemma3InferenceConfig
class RealCfg(Gemma3InferenceConfig):
    @classmethod
    def from_pretrained(cls, model_path, neuron_config, **kw):
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        # force eager-equivalent device path: disable NKI attn/qkv/mlp kernels + fused qkv
        for attr,val in [("attn_kernel_enabled",False),("qkv_kernel_enabled",False),
                         ("qkv_nki_kernel_enabled",False),("qkv_cte_nki_kernel_fuse_rope",False),
                         ("mlp_kernel_enabled",False),("fused_qkv",False),
                         ("attn_tkg_nki_kernel_enabled",False)]:
            try: setattr(neuron_config, attr, val)
            except Exception: pass
        c = cls(neuron_config=neuron_config, load_config=load_pretrained_config(model_path), **kw)
        if getattr(c,"vocab_size",None)!=262144: c.vocab_size=262144
        if getattr(c,"num_global_key_value_heads",None) is None: c.num_global_key_value_heads=1
        try: neuron_config.layer_boundary_markers=True
        except Exception: pass
        return c
