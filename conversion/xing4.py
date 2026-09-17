from __future__ import annotations

import json
import re

from typing import Iterable, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


# Converter for XingChen-AGI/Xing4.0-29B-A4B (HF model_type "xing4_0",
# architecture Xing4_0ForCausalLM), targeting the ROCmFPX fork's XING4 GGUF
# schema (LLM_ARCH_XING4).
#
# Architecture summary:
#   - MLA attention (deepseek2-style decompressed), NOT absorbed
#   - q is always low-rank (q_lora_rank=768); no q_proj
#   - kv_b_proj kept whole as blk.N.attn_kv_b (C++ splits per head)
#   - Per-layer hyper-connections (attn_hc/ffn_hc) on layers 0-39
#   - Layer 40 = MTP/nextn block: full transformer + eh_proj/embed_tokens/
#     enorm/hnorm/shared_head, NO hc tensors
#   - Dense FFN on layers 0..first_k_dense_replace-1, MoE elsewhere
#   - 64 routed experts, 1 shared expert, sigmoid gating with
#     e_score_correction_bias (noaux_tc)
#
# Key differences from DeepSeek V4:
#   - No compressor/recurrent memory → not hybrid
#   - n_embd_head_k(192) != n_embd_head_v(128) → deepseek2 MLA path
#   - No indexer, no hash layers, no o_lora_rank, no compress_ratios
#   - hc tensors named attn_hc.*/ffn_hc.* (not hc_attn_*/hc_ffn_*)
#   - eh_proj is ALREADY combined (do NOT split into e/h)
#   - embed_tokens and shared_head.head DO exist in the MTP layer


@ModelBase.register("Xing4_0ForCausalLM")
class Xing4_0Model(TextModel):
    model_arch = gguf.MODEL_ARCH.XING4
    supports_mtp_export = True

    _main_layers: int = 0
    _nextn_layers: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Enrich hparams from raw config.json, mirroring DeepseekV4Model
        # (ref/deepseek.py:594-612). The base class may narrow hparams to a
        # sub-config; ensure all needed keys are present.
        with open(self.dir_model / "config.json", "r", encoding="utf-8") as config_file:
            raw_hparams = json.load(config_file)
        for key in (
            "first_k_dense_replace",
            "hc_eps",
            "hc_mult",
            "hc_sinkhorn_iters",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "norm_topk_prob",
            "num_nextn_predict_layers",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "kv_lora_rank",
            "routed_scaling_factor",
            "rope_scaling",
            "scoring_func",
            "v_head_dim",
        ):
            if key in raw_hparams:
                self.hparams[key] = raw_hparams[key]

        # Append the nextn/MTP layer(s) to the block list.
        # block_count = num_hidden_layers(40) + num_nextn_predict_layers(1) = 41
        # Mirrors DeepseekV4Model (ref/deepseek.py:639-642).
        self._main_layers = self.hparams["num_hidden_layers"]
        self._nextn_layers = int(self.hparams.get("num_nextn_predict_layers", 0))
        self.block_count = self._main_layers + self._nextn_layers
        self.hparams["num_hidden_layers"] = self.block_count
        self.hparams["n_layers"] = self.block_count
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        # ⭐ SentencePiece, NOT gpt2/BPE. Xing ships a custom slow tokenizer
        # (Xing4_0Tokenizer(PreTrainedTokenizer), auto_map in tokenizer_config.json) over a
        # SentencePiece tokenizer.model — 131072 pieces, exactly vocab_size. There is NO
        # tokenizer.json, so the BPE path does not apply.
        # This also sidesteps a hard failure: _set_vocab_gpt2 -> get_vocab_base calls
        # AutoTokenizer.from_pretrained(dir_model) WITHOUT trust_remote_code
        # (conversion/base.py:1345), transformers doesn't know model_type "xing4_0", builds a
        # generic PreTrainedConfig, and standardize_rope_params() then dies on the missing
        # max_position_embeddings. _create_vocab_sentencepiece reads tokenizer.model directly
        # with SentencePieceProcessor — no AutoTokenizer, no AutoConfig, no remote code.
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            self._set_vocab_gpt2()

        # Emit chat template if present (mirrors DeepseekV4Model, ref/deepseek.py:644-660)
        template_path = self.dir_model / "chat_template.jinja"
        if template_path.is_file():
            self.gguf_writer.add_chat_template(template_path.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])  # 64
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])                # 768

        # Decompressed MLA dims.  key_length/value_length are the per-head
        # dims (K=192, V=128); num_key_value_heads stays 32 (the config value;
        # do NOT force to 1 like the absorbed deepseek2 converter does).
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])                         # 512
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])  # 192
        self.gguf_writer.add_value_length(hparams["v_head_dim"])                            # 128

        # MoE parameters
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])   # 1024
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])                      # 64
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])               # 1
        # noaux_tc: sigmoid scores, top-k with e_score_correction_bias,
        # norm by top-k sum, then scale
        self.gguf_writer.add_expert_weights_scale(hparams.get("routed_scaling_factor", 1.0))  # 2.0
        if hparams.get("norm_topk_prob"):
            self.gguf_writer.add_expert_weights_norm(True)

        # ⭐ Emit the gating function EXPLICITLY. Do not rely on the C++ fallback:
        # deepseek2.cpp:23-31 defaults an absent key to SOFTMAX (its SIGMOID branch is a narrow
        # "n_layer in (47,48) and n_vocab == 154880" GLM-4.7-Lite shape check). Xing is
        # scoring_func="sigmoid" + noaux_tc, so an absent key would run a sigmoid-trained router
        # through softmax — it still loads and generates, which is exactly what makes it dangerous.
        scoring = hparams.get("scoring_func", "sigmoid")
        gating = {
            "sigmoid": gguf.ExpertGatingFuncType.SIGMOID,
            "softmax": gguf.ExpertGatingFuncType.SOFTMAX,
        }.get(scoring)
        if gating is None:
            raise ValueError(f"xing4: unhandled scoring_func {scoring!r}")
        self.gguf_writer.add_expert_gating_func(gating)

        # Dense FFN for leading layers (layers 0-1)
        self.gguf_writer.add_leading_dense_block_count(hparams.get("first_k_dense_replace", 0))  # 2

        # NextN/MTP prediction layers
        if self._nextn_layers:
            self.gguf_writer.add_nextn_predict_layers(self._nextn_layers)  # 1

        # Hyper-connection parameters (ref/deepseek.py:699-702)
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])                    # 4
        self.gguf_writer.add_hyper_connection_sinkhorn_iters(hparams["hc_sinkhorn_iters"]) # 20
        self.gguf_writer.add_hyper_connection_eps(hparams["hc_eps"])                       # 1e-6

        # ⛔ Deliberately NOT add_embedding_length_out(hidden_size * hc_mult).
        # deepseek4 (ref/deepseek.py:702) emits 14336 because ITS hc residual stays expanded all
        # the way to the head — it owns output_hc_{base,fn,scale}. Xing has no output_hc_* tensor:
        # it finalizes by MEAN over the n_hc streams (modeling_xing4_0.py:633), so the hidden state
        # leaving the model is n_embd (3584), not n_embd*hc_mult. Omitting the key is correct —
        # llama_hparams::n_embd_out() returns n_embd when n_embd_out_impl == 0
        # (src/llama-hparams.cpp:91-93). Emitting 14336 would make embedding extraction read
        # 4x too many floats per token.

        # YaRN rope scaling: Xing folds the mscale^2 factor into the
        # attention scale (modeling_xing4_0.py:377-383).  Emit the same
        # legacy yarn log-mul key that DeepseekV2Model emits
        # (ref/deepseek.py:383-387, [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]).
        if (mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * mscale_all)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        # Keep the ORIGINAL name for the base-class fallback: gguf-py/tensor_mapping.py keys are
        # written WITH the prefix ("model.layers.{bid}.mlp.gate.e_score_correction"), so handing
        # super() a stripped name silently disables every fallback entry.
        original = name
        name = name.removeprefix("model.")

        top_level = {
            "embed_tokens.weight": (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "norm.weight":         (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "lm_head.weight":      (gguf.MODEL_TENSOR.OUTPUT, ".weight"),
        }
        if name in top_level:
            tensor, suffix = top_level[name]
            return self.format_tensor_name(tensor, suffix=suffix)

        match = re.match(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            return super().map_tensor_name(original, try_suffixes)

        bid, rest = int(match.group(1)), match.group(2)
        layer_level = {
            # hyper-connections (layers 0-39 only; layer 40 has none)
            "attn_hc.hc_base":  (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "attn_hc.hc_fn":    (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "attn_hc.hc_scale": (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "ffn_hc.hc_base":   (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "ffn_hc.hc_fn":     (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "ffn_hc.hc_scale":  (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            # layernorms
            "input_layernorm.weight":          (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "post_attention_layernorm.weight": (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
            # attention (decompressed MLA)
            "self_attn.q_a_proj.weight":           (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "self_attn.q_a_layernorm.weight":      (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "self_attn.q_b_proj.weight":           (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "self_attn.kv_a_proj_with_mqa.weight": (gguf.MODEL_TENSOR.ATTN_KV_A_MQA, ".weight"),
            "self_attn.kv_a_layernorm.weight":     (gguf.MODEL_TENSOR.ATTN_KV_A_NORM, ".weight"),
            # kv_b kept whole as blk.N.attn_kv_b; C++ splits per head
            # (modeling_xing4_0.py:407-408, same as deepseek2.cpp:108)
            "self_attn.kv_b_proj.weight":          (gguf.MODEL_TENSOR.ATTN_KV_B, ".weight"),
            "self_attn.o_proj.weight":             (gguf.MODEL_TENSOR.ATTN_OUT, ".weight"),
            # router (MoE layers 2-39 and layer 40)
            "mlp.gate.weight":                  (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            # conversion/base.py:578-579 rewrites "e_score_correction_bias" ->
            # "e_score_correction.bias" BEFORE mapping, so the post-rename form is the one
            # that actually arrives here. Both are listed so neither spelling can slip past.
            "mlp.gate.e_score_correction.bias": (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "mlp.gate.e_score_correction_bias": (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            # shared expert
            "mlp.shared_experts.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "mlp.shared_experts.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
            "mlp.shared_experts.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            # dense FFN layers 0..first_k_dense_replace-1
            "mlp.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE, ".weight"),
            "mlp.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP, ".weight"),
            "mlp.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN, ".weight"),
            # stacked expert tensors (produced by modify_tensors)
            "mlp.experts.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE_EXP, ".weight"),
            "mlp.experts.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP_EXP, ".weight"),
            "mlp.experts.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN_EXP, ".weight"),
            # MTP/nextn layer tensors (layer 40 only)
            # eh_proj is ALREADY combined — do NOT split into e/h
            "eh_proj.weight":          (gguf.MODEL_TENSOR.NEXTN_EH_PROJ, ".weight"),
            "embed_tokens.weight":     (gguf.MODEL_TENSOR.NEXTN_EMBED_TOKENS, ".weight"),
            "enorm.weight":            (gguf.MODEL_TENSOR.NEXTN_ENORM, ".weight"),
            "hnorm.weight":            (gguf.MODEL_TENSOR.NEXTN_HNORM, ".weight"),
            "shared_head.norm.weight": (gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM, ".weight"),
            "shared_head.head.weight": (gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD, ".weight"),
        }
        if rest in layer_level:
            tensor, suffix = layer_level[rest]
            return self.format_tensor_name(tensor, bid, suffix=suffix)
        return super().map_tensor_name(original, try_suffixes)

    _experts: list[dict] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Stack per-expert 2-D weights into one 3-D tensor per layer
        # (blk.N.ffn_gate_exps / ffn_up_exps / ffn_down_exps); identical
        # approach to DeepseekV2Model.modify_tensors (ref/deepseek.py:405-432).
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    merged = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(merged, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            pending = [k for d in self._experts for k in d.keys()]
            if len(pending) > 0:
                raise ValueError(f"Unprocessed xing4_0 experts: {pending}")

# OPEN:
# - The tensor name mapping uses MODEL_TENSOR.ATTN_KV_B for the whole kv_b_proj;
#   the prior attempt assumed this enum exists ("inferred from blk.%d.attn_kv_b
#   in the fork's tensor table").  If it does not, the fork's global tensor map
#   (MODEL_ARCH.XING4 via get_tensor_name_map) must cover it.
# - num_key_value_heads is left at the config value (32) for decompressed MLA,
#   NOT forced to 1.  The C++ XING4 loader must read key_length/value_length as
#   per-head dims (192/128) with 32 KV heads, matching the deepseek2 MLA path
#   (not deepseek4, which asserts n_embd_head_k == n_embd_head_v).
# - The merged expert tensor names after stacking are
#   model.layers.{bid}.mlp.experts.{gate,up,down}_proj.weight.  These map to
#   FFN_GATE_EXP/FFN_UP_EXP/FFN_DOWN_EXP via the layer_level dict.  If the
#   base tensor_map already covers this pattern, the layer_level entries are
#   redundant but harmless.
# - add_expert_gating_func("sigmoid") is NOT called because the prior attempt
#   guarded it with hasattr and the reference converter (deepseek.py) does not
#   call it.  The C++ code defaults to SIGMOID when the key is absent.
# - The base class (TextModel.set_gguf_parameters) is assumed to handle
#   rope_theta (10000), rope_scaling type (yarn), and other common hparams
#   automatically.  If not, explicit calls may be needed.
# - Layer 40 (MTP) has its own full transformer (attention + MoE with 64
#   experts + shared expert) plus nextn-specific tensors.  The hc tensors
#   (attn_hc.*/ffn_hc.*) only exist on layers 0-39; layer 40 has none.
#   The tensor_map for 41 blocks will generate hc tensor entries for block 40,
#   but since no model tensors match, they are simply unused.
# - tie_word_embeddings is false in config.json, so both embed_tokens.weight
#   and lm_head.weight are expected to be present.
