#!/usr/bin/env python3
"""Convert MiniCPM-o5 (MiniCPM-o 4.6 family) non-LLM modules to the GGUF files
consumed by llama.cpp-omni.

Input : the joint PyTorch checkpoint (.pt state_dict) + the HF model dir (config.json)
        + an already converted LLM GGUF (for tokenizer KV copy into the TTS GGUF).
Output: <out>/vision/MiniCPM-o5-vision-F16.gguf
        <out>/audio/MiniCPM-o5-audio-F16.gguf      (via convert_apm.py)
        <out>/tts/MiniCPM-o5-tts-F16.gguf
        <out>/tts/MiniCPM-o5-projector-F16.gguf

The LLM itself is converted with convert_hf_to_gguf.py from the HF backbone dir.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402


def f16(t):  return t.to(torch.float16).numpy()
def f32(t):  return t.to(torch.float32).numpy()


def load_sd(pt_path):
    sd = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


# ----------------------------------------------------------------------------- vision
def convert_vision(sd, cfg, pre_cfg, out_path):
    vc = cfg["vision_config"]
    w = gguf.GGUFWriter(out_path, "clip")
    w.add_name("MiniCPM-o5-vision")
    w.add_description("vision encoder (SigLIP + ViT window merger + DownsampleMLP) for MiniCPM-o5")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_bool("clip.has_audio_encoder", False)
    w.add_string("clip.projector_type", "minicpmv4_6")       # -> MiniCPM_o_4_6 in vision.cpp
    w.add_string("general.model_type", "MiniCPM-V-4_6")
    w.add_bool("clip.use_gelu", True)
    w.add_uint32("clip.vision.image_size", int(pre_cfg.get("scale_resolution", cfg.get("image_size", 448))))
    w.add_uint32("clip.vision.patch_size", int(vc["patch_size"]))
    w.add_uint32("clip.vision.embedding_length", int(vc["hidden_size"]))
    w.add_uint32("clip.vision.feed_forward_length", int(vc["intermediate_size"]))
    w.add_uint32("clip.vision.projection_dim", int(cfg["hidden_size"]))
    w.add_uint32("clip.vision.attention.head_count", int(vc["num_attention_heads"]))
    w.add_float32("clip.vision.attention.layer_norm_epsilon", float(vc.get("layer_norm_eps", 1e-6)))
    w.add_uint32("clip.vision.block_count", int(vc["num_hidden_layers"]))
    w.add_uint32("clip.vision.projector.scale_factor", 4)
    w.add_array("clip.vision.wa_layer_indexes", [int(cfg.get("insert_layer_id", 6))])
    w.add_uint32("clip.minicpmv_query_num", int(cfg.get("query_num", 64)))
    w.add_int32("clip.minicpmv_version", 100046)
    mean = pre_cfg.get("norm_mean", pre_cfg.get("image_mean", [0.5, 0.5, 0.5]))
    std  = pre_cfg.get("norm_std",  pre_cfg.get("image_std",  [0.5, 0.5, 0.5]))
    w.add_array("clip.vision.image_mean", [float(x) for x in mean])
    w.add_array("clip.vision.image_std",  [float(x) for x in std])

    def add(name, t, dt=None):
        if dt is None:
            dt = f16 if (t.ndim >= 2 and t.numel() > 4096) else f32
        w.add_tensor(name, dt(t))

    add("v.patch_embd.weight", sd["vpm.embeddings.patch_embedding.weight"])
    add("v.patch_embd.bias",   sd["vpm.embeddings.patch_embedding.bias"])
    add("v.position_embd.weight", sd["vpm.embeddings.position_embedding.weight"])
    add("v.post_ln.weight", sd["vpm.post_layernorm.weight"])
    add("v.post_ln.bias",   sd["vpm.post_layernorm.bias"])
    n_layer = int(vc["num_hidden_layers"])
    amap = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "out_proj": "attn_out"}
    for il in range(n_layer):
        p = f"vpm.encoder.layers.{il}."
        for hf, gg in amap.items():
            add(f"v.blk.{il}.{gg}.weight", sd[p + f"self_attn.{hf}.weight"])
            add(f"v.blk.{il}.{gg}.bias",   sd[p + f"self_attn.{hf}.bias"])
        add(f"v.blk.{il}.ln1.weight", sd[p + "layer_norm1.weight"])
        add(f"v.blk.{il}.ln1.bias",   sd[p + "layer_norm1.bias"])
        add(f"v.blk.{il}.ln2.weight", sd[p + "layer_norm2.weight"])
        add(f"v.blk.{il}.ln2.bias",   sd[p + "layer_norm2.bias"])
        add(f"v.blk.{il}.ffn_up.weight",   sd[p + "mlp.fc1.weight"])
        add(f"v.blk.{il}.ffn_up.bias",     sd[p + "mlp.fc1.bias"])
        add(f"v.blk.{il}.ffn_down.weight", sd[p + "mlp.fc2.weight"])
        add(f"v.blk.{il}.ffn_down.bias",   sd[p + "mlp.fc2.bias"])
    # ViT window-attention merger (inserted after layer insert_layer_id)
    m = "vit_merger."
    add("v.vit_merger.ln1.weight", sd[m + "layer_norm1.weight"])
    add("v.vit_merger.ln1.bias",   sd[m + "layer_norm1.bias"])
    for hf, gg in amap.items():
        add(f"v.vit_merger.{gg}.weight", sd[m + f"self_attn.{hf}.weight"])
        add(f"v.vit_merger.{gg}.bias",   sd[m + f"self_attn.{hf}.bias"])
    add("v.vit_merger.ds_ln.weight",       sd[m + "pre_norm.weight"])
    add("v.vit_merger.ds_ln.bias",         sd[m + "pre_norm.bias"])
    add("v.vit_merger.ds_ffn_up.weight",   sd[m + "linear_1.weight"])
    add("v.vit_merger.ds_ffn_up.bias",     sd[m + "linear_1.bias"])
    add("v.vit_merger.ds_ffn_down.weight", sd[m + "linear_2.weight"])
    add("v.vit_merger.ds_ffn_down.bias",   sd[m + "linear_2.bias"])
    # final DownsampleMLP merger (resampler.mlp.0.*) -> LLM hidden
    r = "resampler.mlp.0."
    add("mm.input_norm.weight", sd[r + "pre_norm.weight"])
    add("mm.input_norm.bias",   sd[r + "pre_norm.bias"])
    add("mm.up.weight",   sd[r + "mlp.0.weight"])
    add("mm.up.bias",     sd[r + "mlp.0.bias"])
    add("mm.down.weight", sd[r + "mlp.2.weight"])
    add("mm.down.bias",   sd[r + "mlp.2.bias"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("vision ->", out_path, os.path.getsize(out_path) // 2**20, "MiB")


# ----------------------------------------------------------------------------- audio
def convert_audio(sd, cfg, model_dir, out_dir, python):
    tmp = os.path.join(out_dir, "_apm_tmp"); os.makedirs(tmp, exist_ok=True)
    keys = [k for k in sd if k.startswith(("apm.", "audio_projection_layer."))]
    torch.save({k: sd[k].float() for k in keys}, os.path.join(tmp, "minicpmo.whisper"))
    shutil.copy2(os.path.join(model_dir, "config.json"), os.path.join(tmp, "config.json"))
    subprocess.check_call([python, os.path.join(HERE, "convert_apm.py"), tmp,
                           os.path.join(tmp, "minicpmo.whisper"), out_dir],
                          env={**os.environ, "PYTHONPATH": os.path.join(HERE, "..", "..", "..", "gguf-py")})
    for f in os.listdir(out_dir):
        if f == "minicpmo-apm.gguf":
            shutil.move(os.path.join(out_dir, f), os.path.join(out_dir, "MiniCPM-o5-audio-F16.gguf"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("audio ->", os.listdir(out_dir))


# ----------------------------------------------------------------------------- tts
def copy_tokenizer_kv(w, llm_gguf):
    r = gguf.GGUFReader(llm_gguf)
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            et = f.types[1]
            if et == gguf.GGUFValueType.STRING:
                vals = [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]
            else:
                vals = [f.parts[i].tolist()[0] for i in f.data]
            w.add_array(k, vals)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(k, bytes(f.parts[f.data[0]]).decode("utf-8"))
        elif t == gguf.GGUFValueType.BOOL:
            w.add_bool(k, bool(f.parts[f.data[0]][0]))
        elif t in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32):
            v = int(f.parts[f.data[0]][0])
            (w.add_uint32 if t == gguf.GGUFValueType.UINT32 else w.add_int32)(k, v)
        elif t == gguf.GGUFValueType.FLOAT32:
            w.add_float32(k, float(f.parts[f.data[0]][0]))
        else:
            print("skip tokenizer kv", k, t)


def convert_tts(sd, cfg, llm_gguf, out_path, proj_path):
    tc = cfg["tts_config"]
    H, L, NH = int(tc["hidden_size"]), int(tc["num_hidden_layers"]), int(tc["num_attention_heads"])
    w = gguf.GGUFWriter(out_path, "llama")
    w.add_name("MiniCPM-o5-tts")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_block_count(L)
    w.add_context_length(int(tc.get("max_position_embeddings", 4096)))
    w.add_embedding_length(H)
    w.add_feed_forward_length(int(tc["intermediate_size"]))
    w.add_head_count(NH)
    w.add_head_count_kv(int(tc.get("num_key_value_heads", NH)))
    w.add_rope_freq_base(float(tc.get("rope_theta", 10000.0)))
    w.add_layer_norm_rms_eps(float(tc.get("rms_norm_eps", 1e-6)))
    w.add_key_length(H // NH); w.add_value_length(H // NH)
    w.add_rope_dimension_count(H // NH)
    w.add_vocab_size(int(sd["tts.emb_text.weight"].shape[0]))  # token_embd == emb_text (LLM vocab), as in the o4.5 TTS GGUF
    # o5 extras as metadata so omni.cpp can size buffers without hardcoding
    w.add_uint32("minicpmtts.llm_hidden_size", int(cfg["hidden_size"]))
    w.add_uint32("minicpmtts.num_audio_tokens", int(sd["tts.emb_code.0.weight"].shape[0]))
    w.add_uint32("minicpmtts.num_text_tokens", int(sd["tts.emb_text.weight"].shape[0]))
    w.add_string("minicpmtts.condition_type", tc.get("condition_type", "tokens_and_hidden_merge"))
    w.add_bool("minicpmtts.normalize_projected_hidden", bool(tc.get("normalize_projected_hidden", False)))
    w.add_string("minicpmtts.attention_type", tc.get("attention_type", "full_attention"))
    w.add_uint32("minicpmtts.audio_bos_token_id", int(tc.get("audio_bos_token_id", 0)))
    w.add_uint32("minicpmtts.text_eos_token_id", int(tc.get("text_eos_token_id", 0)))
    copy_tokenizer_kv(w, llm_gguf)

    def add(name, t):
        w.add_tensor(name, f16(t) if (t.ndim >= 2 and t.numel() > 4096) else f32(t))

    add("token_embd.weight", sd["tts.emb_text.weight"])  # text side of the TTS input space; audio tokens enter as embeddings
    add("output_norm.weight", sd["tts.model.norm.weight"])
    lm = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
          "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
          "mlp.down_proj": "ffn_down", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
    def permute_qk(t, n_head):
        # llama.cpp LLAMA arch expects Q/K rows permuted from HF's (head, 2, half) rotary layout
        # to interleaved pairs; identical to convert_hf_to_gguf.py LlamaModel.permute
        shp = t.shape
        return t.reshape(n_head, 2, shp[0] // n_head // 2, *shp[1:]).swapaxes(1, 2).reshape(shp)
    n_kv = int(tc.get("num_key_value_heads", NH))
    for il in range(L):
        for hf, gg in lm.items():
            t = sd[f"tts.model.layers.{il}.{hf}.weight"]
            if gg == "attn_q": t = permute_qk(t, NH)
            if gg == "attn_k": t = permute_qk(t, n_kv)
            add(f"blk.{il}.{gg}.weight", t)
    add("emb_code.0.weight", sd["tts.emb_code.0.weight"])
    add("emb_text.weight",   sd["tts.emb_text.weight"])
    g = sd["tts.head_code.0.parametrizations.weight.original0"].float()
    v = sd["tts.head_code.0.parametrizations.weight.original1"].float()
    head = g * v / (v.norm(dim=1, keepdim=True) + 1e-12)          # (num_audio_tokens, hidden)
    # omni.cpp expects head_code data hidden-major ([768][6562] row-major, as the o4.5 converter wrote it)
    add("head_code.0.weight", head.T.contiguous())
    for p in ("projector_semantic", "projector_spk"):
        for n in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
            add(f"{p}.{n}", sd[f"tts.{p}.{n}"])
        for n in ("weight", "bias"):
            k = f"tts.{p}_norm.{n}"
            if k in sd: add(f"{p}_norm.{n}", sd[k])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("tts ->", out_path, os.path.getsize(out_path) // 2**20, "MiB")

    # stand-alone projector_semantic GGUF (format expected by omni.cpp projector_init)
    pw = gguf.GGUFWriter(proj_path, "projector")
    pw.add_name("MiniCPM-o5-projector-semantic")
    pw.add_uint32("projector.in_dim", int(cfg["hidden_size"]))
    pw.add_uint32("projector.out_dim", H)
    for n in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
        pw.add_tensor(n, f32(sd[f"tts.projector_semantic.{n}"]))
    for n in ("weight", "bias"):
        k = f"tts.projector_semantic_norm.{n}"
        if k in sd: pw.add_tensor(f"norm.{n}", f32(sd[k]))
    pw.write_header_to_file(); pw.write_kv_data_to_file(); pw.write_tensors_to_file(); pw.close()
    print("projector ->", proj_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="joint checkpoint .pt")
    ap.add_argument("--model-dir", required=True, help="HF dir with config.json / preprocessor_config.json")
    ap.add_argument("--llm-gguf", required=True, help="converted LLM GGUF (tokenizer source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apm-python", default=sys.executable, help="python with librosa for convert_apm.py")
    ap.add_argument("--only", default="vision,audio,tts")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    pre = {}
    pp = os.path.join(a.model_dir, "preprocessor_config.json")
    if os.path.exists(pp): pre = json.load(open(pp))
    sd = load_sd(a.pt)
    only = set(a.only.split(","))
    for d in ("vision", "audio", "tts"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    if "vision" in only:
        convert_vision(sd, cfg, pre, os.path.join(a.out, "vision", "MiniCPM-o5-vision-F16.gguf"))
    if "audio" in only:
        convert_audio(sd, cfg, a.model_dir, os.path.join(a.out, "audio"), a.apm_python)
    if "tts" in only:
        convert_tts(sd, cfg, a.llm_gguf, os.path.join(a.out, "tts", "MiniCPM-o5-tts-F16.gguf"),
                    os.path.join(a.out, "tts", "MiniCPM-o5-projector-F16.gguf"))


if __name__ == "__main__":
    main()
