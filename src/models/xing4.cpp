// llama_model_xing4 — Xing4.0-29B-A4B (HF model_type "xing4_0", arch "xing4").
//
// Two existing implementations are combined, per the verified fact sheet:
//   * attention / MoE / output head: llama_model_deepseek2 — Xing's attention
//     tensor set is identical (wq_a + q_a_norm + wq_b, wkv_a_mqa + kv_a_norm +
//     fused wkv_b split per head in the graph, dense wo; there is no
//     self_attn.q_proj, q is always low-rank), and its head dims differ
//     (K = qk_nope 128 + qk_rope 64 = 192, V = 128), which is exactly what
//     deepseek4's `n_embd_head_k == n_embd_head_v` assert rejects.
//   * hyper-connections + MTP: llama_model_deepseek4 — the same
//     ggml_dsv4_hc_split_sinkhorn / _weighted_sum / _expand ops, called the
//     same way.
//
// Xing-specific structure, all read off the real weights:
//   * the residual is [n_embd, n_hc, n_tokens]; hc wraps both the attention
//     and the FFN block on layers 0..n_layer-nextn_predict_layers-1
//   * the MTP block (the last block) ships NO hc tensors at all — it is a
//     plain deepseek-style MTP layer over the merged hidden, with a combined
//     nextn.eh_proj plus nextn.embed_tokens/enorm/hnorm/shared_head_norm/
//     shared_head_head
//   * there are no output_hc_* tensors: the head is a mean over the n_hc
//     streams followed by output_norm and lm_head
//   * there is no clamp anywhere — ggml_dsv4_hc_split_sinkhorn takes no clamp
//     argument; HF's mhc_h_res_clamp_min/max is a training-time guard only

#include "models.h"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace {

// same payload as dsv4_hc_mix in deepseek4-graph.cpp
struct xing4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

// hyper-connection pre-mix, verbatim from dsv4_hc_pre (deepseek4-graph.cpp):
// Xing4_0HyperConnection does unweighted rms-norm over the flattened streams,
// projects with hc_fn, then scales/shifts with hc_scale/hc_base and runs
// Sinkhorn on the comb block — all of which the fused op owns
static xing4_hc_mix xing4_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int            sinkhorn_iters,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat); // [mix_hc, n_tokens]
    // Xing's Sinkhorn puts eps in the denominator and applies none to `pre`; the dsv4
    // entry point adds it to the value instead, which inflates legitimately-tiny entries
    // (~1e-7) by 5-17x and Sinkhorn then amplifies that floor. See ggml_xing4_* variant.
    ggml_tensor * split = ggml_xing4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]
    // ggml_dsv4_hc_expand contracts ne1 and indexes the destination stream by ne0, so it
    // wants [dst_hc, src_hc]; the Sinkhorn writes dst*n_hc + src, which reshapes to
    // [src_hc, dst_hc] (HF's comb[dst][src]). Swap the two hc axes so the op contracts the
    // axis HF contracts. Verified arithmetically at layer 0: convention A reproduces HF's
    // +0.0106, convention B reproduces the unpatched +(-0.0031). See patch header.
    comb = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

// block output folded back into the streams, verbatim from dsv4_hc_post:
// post * block_out + comb @ streams
static ggml_tensor * xing4_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == n_hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == n_hc);
    GGML_ASSERT(comb->ne[1] == n_hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

// mean over the hyper-connection streams: [n_embd, n_hc, n_tokens]
// -> [n_embd, n_tokens]. Xing's output head (modeling_xing4_0.py:633).
// Same code as dsv4_hc_mean in deepseek4-graph.cpp.
static ggml_tensor * xing4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n_hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t ihc = 1; ihc < n_hc; ++ihc) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], ihc * x->nb[1]));
    }

    return ggml_scale(ctx, acc, 1.0f / n_hc);
}

} // namespace

// the graph class is TU-local: models.h declares llama_model_xing4 with the
// three overrides only, so (unlike deepseek2/qwen4exp) there is no nested
// `graph` type to hang methods off
namespace {

class llm_build_xing4 : public llm_graph_context {
public:
    llm_build_xing4(const llama_model & model, const llm_graph_params & params);

private:
    // deepseek2 does all of its work inside the ctor body, so `model` is simply the ctor
    // parameter there. This class splits the block into helpers, so it has to hold the
    // reference itself for them to reach model.layers[il].
    const llama_model & model;

    // decompressed MLA over the normed block input, dense wo on the output
    ggml_tensor * build_mla_attn(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn_kv,
                                 ggml_tensor * inp_pos, float kq_scale, int il);
    // dense FFN on the leading layers, MoE + shared expert elsewhere
    ggml_tensor * build_layer_ffn(ggml_tensor * cur, int il);
};

} // namespace

void llama_model_xing4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false); // first_k_dense_replace
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    // head sizes after decompressing wkv_b; read optionally (as deepseek2
    // does) so GGUFs carrying only attention.key_length/value_length fall
    // back to those
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,          hparams.n_hc); // hc_mult
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, hparams.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPS,            hparams.hc_eps);

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        // Xing scores with sigmoid + the noaux_tc correction bias
        // (config.json "scoring_func"/"topk_method", modeling_xing4_0.py:155)
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script; the graph rebuilds
        // YaRN's mscale^2 attention scale from it
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_xing4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t kv_lora_rank    = hparams.n_lora_kv;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_hc            = hparams.n_hc;
    const int64_t hc_dim          = n_hc * n_embd;
    const int64_t hc_mix          = (2 + n_hc) * n_hc; // pre + post + comb (hc*hc)

    // actual head sizes you get after decompressing wkv_b (deepseek2.cpp:61-66)
    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // no output_hc_* tensors: Xing finalizes by mean over the hc streams
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // the MTP block is the last nextn_predict_layers blocks; it carries no
        // attn_hc.*/ffn_hc.* tensors — hyper-connections stop at the block
        // before it
        const bool is_nextn = hparams.nextn_predict_layers > 0 &&
            static_cast<uint32_t>(i) >= hparams.n_layer - hparams.nextn_predict_layers;

        if (!is_nextn) {
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix}, 0);
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);
        }

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        // norms the kv latent only, not the rope tail
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        // fused kv_b_proj, kept whole ({kv_lora_rank, n_head*(nope+v)}); the
        // graph splits it per head
        layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            // layer 40 (the MTP block) is MoE too — it has its own full router
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }

        if (is_nextn) {
            // combined eh_proj (the checkpoint ships a single eh_proj, not
            // split e_proj/h_proj), plus the MTP's own embedding and head
            // {2*n_embd, n_embd}: ne0 is the INPUT width (the [e ; h] concat, 7168), ne1 the
            // output (3584) — HF Linear(in=2*hidden, out=hidden) transposes to this in GGUF.
            // Verified against the file: blk.40.nextn.eh_proj.weight = [7168, 3584].
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, 0);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, 0);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, 0);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, 0);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, 0);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_xing4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        GGML_ASSERT(hparams.nextn_predict_layers > 0 && "Xing4.0 MTP graph requires an appended prediction layer");
    }
    return std::make_unique<llm_build_xing4>(*this, params);
}

// trunk: broadcast the embedding over the hc streams, run layers
// 0..n_main_layers-1 with hc around both blocks, finalize by mean over the
// streams. MTP (LLM_GRAPH_TYPE_DECODER_MTP): run only the last block, as a
// plain deepseek-style layer over the merged hidden.
llm_build_xing4::llm_build_xing4(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params), model(model) {
    const int64_t n_hc = hparams.n_hc;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(hparams.n_lora_q > 0);
    GGML_ASSERT(hparams.n_lora_kv > 0);

    const bool is_mtp = params.gtype == LLM_GRAPH_TYPE_DECODER_MTP;
    const int n_main_layers = n_layer - (int) hparams.nextn_predict_layers;
    GGML_ASSERT(n_main_layers > 0);
    if (is_mtp) {
        GGML_ASSERT(hparams.nextn_predict_layers == 1 && "Xing4.0 MTP currently supports one appended prediction layer");
    }

    // K per head after decompression; != V per head, which is why this is not
    // deepseek4's attention
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();

    // We have to pre-scale kq_scale to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k_mla));

    ggml_tensor * inpL;

    if (is_mtp) {
        const int il = n_main_layers;
        const auto & layer = model.layers[il];
        GGML_ASSERT(layer.nextn.eh_proj          && "Xing4.0 MTP block missing nextn.eh_proj");
        GGML_ASSERT(layer.nextn.embed_tokens     && "Xing4.0 MTP block missing nextn.embed_tokens");
        GGML_ASSERT(layer.nextn.enorm            && "Xing4.0 MTP block missing nextn.enorm");
        GGML_ASSERT(layer.nextn.hnorm            && "Xing4.0 MTP block missing nextn.hnorm");
        GGML_ASSERT(layer.nextn.shared_head_norm && "Xing4.0 MTP block missing nextn.shared_head_norm");
        GGML_ASSERT(layer.nextn.shared_head_head && "Xing4.0 MTP block missing nextn.shared_head_head");

        // the MTP block carries no hc tensors, so it consumes the merged
        // hidden: width n_embd, i.e. the default hparams.n_embd_pre_norm()
        auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd);
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tokens);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->h);
        ggml_set_name(inp->h, "mtp_h_input");

        ggml_tensor * inp_tokens = inp->tokens;

        // e side: embed the token, norm at n_embd width
        ggml_tensor * tok_embd = ggml_get_rows(ctx0, layer.nextn.embed_tokens, inp_tokens);
        tok_embd = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(tok_embd, "mtp_enorm", il);

        // h side: the target's merged hidden
        ggml_tensor * h_embd = build_norm(inp->h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(h_embd, "mtp_hnorm", il);

        // eh_proj @ [e ; h] — embedding half first
        ggml_tensor * eh = ggml_concat(ctx0, tok_embd, h_embd, 0);
        cb(eh, "mtp_concat", il);

        inpL = ggml_mul_mat(ctx0, layer.nextn.eh_proj, eh);
        cb(inpL, "mtp_inp", il);

        res->add_input(std::move(inp));
    } else {
        inpL = build_inp_embd(model.tok_embd);
        // replicate the embedding over the hc streams
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn_kv = build_attn_inp_kv();

    const int il_begin = is_mtp ? n_main_layers : 0;
    const int il_end   = is_mtp ? n_layer : n_main_layers;
    for (int il = il_begin; il < il_end; ++il) {
        if (!is_mtp && (size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = xing4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * cur;
        xing4_hc_mix mix {};

        if (!is_mtp) {
            mix = xing4_hc_pre(ctx0, inpL,
                    layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                    n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            cur = mix.x;
            cb(cur, "hc_attn_pre", il);
            cb(mix.mixes, "hc_attn_pre_mixes", il);
            cb(mix.pre, "hc_attn_pre_weights", il);
            cb(mix.post, "hc_attn_pre_post_weights", il);
            cb(mix.comb, "hc_attn_pre_comb", il);
        } else {
            // the MTP block has no hc tensors: it runs on its input directly
            cur = inpL;
        }

        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_mla_attn(cur, inp_attn_kv, inp_pos, kq_scale, il);

        if (!is_mtp) {
            inpL = xing4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
            cb(inpL, "hc_attn_post", il);
        } else {
            inpL = ggml_add(ctx0, cur, residual);
        }

        residual = inpL;
        if (!is_mtp) {
            mix = xing4_hc_pre(ctx0, inpL,
                    layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                    n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            cur = mix.x;
            cb(cur, "hc_ffn_pre", il);
            cb(mix.mixes, "hc_ffn_pre_mixes", il);
            cb(mix.pre, "hc_ffn_pre_weights", il);
            cb(mix.post, "hc_ffn_pre_post_weights", il);
            cb(mix.comb, "hc_ffn_pre_comb", il);
        } else {
            cur = inpL;
        }

        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);

        if (!is_mtp) {
            inpL = xing4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
            cb(inpL, "hc_ffn_post", il);
        } else {
            inpL = ggml_add(ctx0, cur, residual);
        }
    }

    if (!is_mtp && (size_t) il_end < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il_end]) {
        res->t_layer_inp[il_end] = xing4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[il_end], "layer_inp", il_end);
        ggml_build_forward_expand(gf, res->t_layer_inp[il_end]);
    }

    ggml_tensor * cur;
    if (!is_mtp) {
        // no output_hc_* tensors: collapse the streams, then the output norm
        cur = xing4_hc_mean(ctx0, inpL);
        cb(cur, "result_hc_mean", -1);

        if (cparams.embeddings_pre_norm) {
            // what the MTP block consumes as its h input (width n_embd)
            res->t_h_pre_norm = cur;
            cb(res->t_h_pre_norm, "h_pre_norm", -1);
        }
    } else {
        cur = inpL;
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    const auto & head_layer = model.layers[n_main_layers];
    cur = build_norm(cur,
            is_mtp ? head_layer.nextn.shared_head_norm : model.output_norm,
            nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // the MTP block ships its own head copy
    cur = ggml_mul_mat(ctx0, is_mtp ? head_layer.nextn.shared_head_head : model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Decompressed MLA, from the non-absorbed branch of llama_model_deepseek2's
// graph: q through wq_a -> q_a_norm -> wq_b; the latent through
// wkv_a_mqa -> kv_a_norm -> wkv_b split per head into K-nope and V; dense wo.
// `cur` in is the normed [n_embd, n_tokens] block input, the return is
// [n_embd, n_tokens]. K head = nope + rope (192) != V head (128).
ggml_tensor * llm_build_xing4::build_mla_attn(
        ggml_tensor * cur,
        llm_graph_input_attn_kv * inp_attn_kv,
        ggml_tensor * inp_pos,
        float         kq_scale,
        int           il) {
    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v       = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;

    ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
    cb(q, "q_lora", il);

    q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "q_lora_norm", il);

    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
    cb(q, "q", il);

    // split into {n_embd_head_qk_nope, n_head, n_tokens}
    ggml_tensor * q_nope =
        ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                     ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    cb(q_nope, "q_nope", il);

    // and {n_embd_head_qk_rope, n_head, n_tokens}
    ggml_tensor * q_pe = ggml_view_3d(
        ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
        ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_pe, "q_pe", il);

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "kv_cmpr_pe", il);

    // split into {kv_lora_rank, n_tokens}
    ggml_tensor * kv_cmpr =
        ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                     ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "kv_cmpr", il);

    // and {n_embd_head_qk_rope, 1, n_tokens}
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(k_pe, "k_pe", il);

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    // decompress the latent into per-head K-nope and V
    ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
    cb(kv, "kv", il);

    // split into {n_embd_head_qk_nope, n_head, n_tokens}
    ggml_tensor * k_nope =
        ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                     ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                     ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
    cb(k_nope, "k_nope_view", il);

    // and {n_embd_head_v, n_head, n_tokens}
    ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                      ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                      ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                      ggml_row_size(kv->type, n_embd_head_qk_nope));
    cb(Vcur, "Vcur_view", il);

    Vcur = ggml_cont(ctx0, Vcur);
    cb(Vcur, "Vcur_cont", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(Qcur, "Qcur", il);

    ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
    cb(Kcur, "Kcur", il);

    // note: MLA without the absorption optimization converts into MHA (ie: GQA with full n_head groups)
    cur = build_attn(inp_attn_kv,
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    return cur;
}

// dense FFN on the leading layers, MoE + shared expert elsewhere
// (modeling_xing4_0.py:493-496)
ggml_tensor * llm_build_xing4::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
        return cur;
    }

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            layer.ffn_gate_up_exps);
    cb(moe_out, "ffn_moe_out", il);

    // FFN shared expert
    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "ffn_out", il);

    return cur;
}

// OPEN:
// * attention KV layout. load_arch_tensors creates the FUSED blk.%d.attn_kv_b
//   (deepseek2.cpp:108), not the MLA-split attn_k_b/attn_v_b pair — SPEC §2
//   cites deepseek2's wkv_b line for xing4's attention tensor set, and the HF
//   checkpoint ships a fused self_attn.kv_b_proj in all 41 blocks. If the
//   converter ever emits the split pair instead, deepseek2.cpp:104-109's
//   is_mla branch (wk_b {nope, kv_lora_rank, n_head} + wv_b
//   {kv_lora_rank, v, n_head} plus the absorbed path at deepseek2.cpp:291-329)
//   is what has to be added here; nothing in this file covers it.
// * head-size KV keys. n_embd_head_k_mla()/n_embd_head_v_mla() are read from
//   attention.key_length_mla/value_length_mla when present, else they fall
//   back to attention.key_length/value_length. I could not verify which pair
//   the xing4 converter writes; both resolve to 192/128 for a correct GGUF.
// * res->t_h_pre_norm (trunk, under cparams.embeddings_pre_norm, mirroring
//   deepseek4) is the merged hidden BEFORE output_norm. HF feeds the MTP h
//   input with the hidden AFTER model.norm. If the fork's MTP runner copies
//   t_h_pre_norm into the draft's inp->h, xing4's h differs from HF by that
//   final rms-norm (hnorm is unweighted, so it removes the scale but not the
//   weight). Same convention-vs-HF question as deepseek4 itself.
// * the MTP graph's h input is created n_embd wide, on the assumption that
//   the runner sizes it from hparams.n_embd_pre_norm(), which the integrator
//   left at the default n_embd for xing4 (deepseek4 overrides it to
//   n_embd*n_hc because ITS mtp eats expanded streams).
// * res->t_h_pre_norm is NOT exported from the MTP graph. deepseek4 does not
//   either; qwen4exp does (it needs the wide streams for its next draft step).
//   With nextn_predict_layers == 1 nothing should consume it.
// * res->t_layer_inp[il] (per-layer hidden states, merged) is populated the
//   way deepseek4 does it. Nothing in xing4 consumes it; the blocks are dead
//   unless the model is run as an embedding-output model.
// * build_cvec is not applied per layer (deepseek4 omits it too): the layer
//   output is the [n_embd, n_hc, n_tokens] stream tensor, which a
//   [n_embd, n_tokens] control vector cannot broadcast onto. Control vectors
//   therefore do not work for this arch.
// * type stays LLM_TYPE_UNKNOWN: no LLM_TYPE_* matching 29B-A4B was verified,
//   so none was invented.
// * layer.nextn.shared_head_head is used for the MTP logits. HF ties it to
//   lm_head, so it equals model.output for this checkpoint.
