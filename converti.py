"""
Passo 1 - da eseguire UNA volta sola, con le altre applicazioni chiuse.

Converte il transformer di FILE2 (formato ComfyUI) in formato diffusers
rinominando le chiavi (mappa ricavata dal confronto delle forme),
poi lo salva in OUT, pronto per essere caricato con from_pretrained.
"""
import gc
import os
import sys

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from diffusers import LTX2VideoTransformer3DModel

REPO = r"C:\Users\User\.cache\huggingface\hub\models--Lightricks--LTX-2.5-Diffusers\snapshots\a97165959b05f5eb52a9f40f19c55957e334abe9"
FILE2 = r".\model\ltx-2.5-nsfw-step7000.safetensors"
OUT = r"G:\realTime_create\model\transformer_nsfw"
PFX = "model.diffusion_model."

# 1) mappa di rinominamento (prefissi di primo livello)
TOP = {
    "adaln_single.": "time_embed.",
    "audio_adaln_single.": "audio_time_embed.",
    "patchify_proj.": "proj_in.",
    "audio_patchify_proj.": "audio_proj_in.",
    "prompt_adaln_single.": "prompt_adaln.",
    "audio_prompt_adaln_single.": "audio_prompt_adaln.",
    "av_ca_a2v_gate_adaln_single.": "av_cross_attn_video_a2v_gate.",
    "av_ca_audio_scale_shift_adaln_single.": "av_cross_attn_audio_scale_shift.",
    "av_ca_v2a_gate_adaln_single.": "av_cross_attn_audio_v2a_gate.",
    "av_ca_video_scale_shift_adaln_single.": "av_cross_attn_video_scale_shift.",
}


def fix(k):
    for old, new in TOP.items():
        if k.startswith(old):
            return new + k[len(old):]
    if k.startswith("transformer_blocks."):
        k = k.replace(".k_norm.", ".norm_k.").replace(".q_norm.", ".norm_q.")
        if k.endswith(".scale_shift_table_a2v_ca_audio"):
            k = k[: -len("scale_shift_table_a2v_ca_audio")] + "audio_a2v_cross_attn_scale_shift_table"
        elif k.endswith(".scale_shift_table_a2v_ca_video"):
            k = k[: -len("scale_shift_table_a2v_ca_video")] + "video_a2v_cross_attn_scale_shift_table"
    return k


# 1b) controllo preventivo dei tipi di dato (legge solo le intestazioni)
dts = {}
with safe_open(FILE2, "pt") as sf:
    for k in sf.keys():
        d = sf.get_slice(k).get_dtype()
        dts[d] = dts.get(d, 0) + 1
print("tipi di dato in FILE2:", dts)
if any(d in dts for d in ("I8", "U8", "F8_E4M3", "F8_E5M2")):
    sys.exit(
        "FILE2 contiene pesi quantizzati (int8/fp8): la conversione darebbe valori sbagliati.\n"
        "Incollami la riga 'tipi di dato' qui sopra."
    )

# 2) pesi di FILE2 senza prefisso, in bf16
print("Carico FILE2 in RAM (circa 38 GB)...")
sd = {}
with safe_open(FILE2, "pt") as sf:
    for k in sf.keys():
        t = sf.get_tensor(k)
        if t.is_floating_point():
            t = t.to(torch.bfloat16)
        sd[k.removeprefix(PFX)] = t

# 3) rinominamento delle chiavi
sd = {fix(k): v for k, v in sd.items()}

# 4) modello vuoto dalla config del repo e confronto di chiavi e forme
cfg = LTX2VideoTransformer3DModel.load_config(os.path.join(REPO, "transformer"))
with init_empty_weights():
    model = LTX2VideoTransformer3DModel.from_config(cfg)
expected = model.state_dict()

missing = sorted(set(expected) - set(sd))
extra = sorted(set(sd) - set(expected))
bad_shape = sorted(
    k for k in set(sd) & set(expected) if tuple(sd[k].shape) != tuple(expected[k].shape)
)

print("mancanti      :", len(missing), missing[:10])
print("in piu'       :", len(extra), extra[:10], "(verranno scartati)")
print("forma diversa :", len(bad_shape), bad_shape[:10])

if missing or bad_shape:
    sys.exit("Conversione incompleta: incollami le tre righe qui sopra.")

# 5) carico e salvo
sd = {k: v for k, v in sd.items() if k in expected}
model.load_state_dict(sd, assign=True, strict=True)
del sd
gc.collect()

model.save_pretrained(OUT, max_shard_size="5GB")
print("fatto ->", OUT)