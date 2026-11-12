# Stage 0 at 4-bit

`stage0_4bit.py` turns the hybrid checkpoint of cebeuq's recipe into the checkpoint the current form serves: on
layers 0-29 (the Blackwell stage of the two-host layout) the attention `q_b_proj` and `o_proj` and the shared
experts are re-quantized from their BF16 parents to NVFP4 (W4A16, run by vLLM's Marlin NVFP4 kernel) instead of the
INT8-g128 residue. `fused_qkv_a_proj` (q_a + kv_a), `kv_b_proj` and the dense MLPs of layers 0-2 stay INT8: at 4-bit
they add more error, or save no bytes during decode. Layers 30-77 and the routed experts are unchanged.

**No weights are distributed here.** The tool builds the new checkpoint on your machine from two public sources.

## Inputs

1. **The hybrid checkpoint** (NVFP4 routed experts plus an INT8-g128 residue), converted with cebeuq's requant tool
   from dealignai's abliterated NVFP4 checkpoint, as his recipe describes:
   [cebeuq/glm53-cmp170hx-recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe). The tool reads it and never
   writes to it.
2. **The BF16 parents of the residue.** dealignai's checkpoint, now published as
   [dealignai/GLM-5.3-UNCENSORED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-NVFP4) (the older name
   dealignai/GLM-5.3-ABLITERATED-NVFP4 holds only a README), keeps every attention, shared-expert and layer 0-2 MLP
   tensor in BF16. The residue of the hybrid checkpoint is the INT8 quantization of exactly these tensors, so the
   4-bit tensors are quantized from the BF16 parents, not from the INT8. The tool needs the index and the 34 shards
   that hold layers 0-29:

```bash
REV=409ddf484d16cb7b13d390e8b30942890164d5af   # the revision the served checkpoint was built from
hf download dealignai/GLM-5.3-UNCENSORED-NVFP4 --revision $REV --local-dir hub model.safetensors.index.json \
  $(for i in 1 2 5 9 12 13 16 20 24 27 31 35 38 42 46 49 50 53 57 61 64 68 72 75 79 119 120 160 200 201 241 278 281 282; \
    do printf 'model-%05d-of-00282.safetensors ' $i; done)
```

`--dry-run` prints the same list from the index.

## Run

The recipe's pinned engine image has the libraries the tool uses (torch 2.13, safetensors 0.8.0, compressed-tensors
0.17.0). It runs on the CPU and holds one shard at a time.

```bash
docker run --rm --network none -e OMP_NUM_THREADS=8 \
  -v /srv/glm53/models:/srv/glm53/models -v $PWD:/work --entrypoint python3 \
  lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e -I /work/stage0_4bit.py \
  --model /srv/glm53/models/glm53-abliterated-nvfp4e-int8r \
  --hub /work/hub \
  --out /srv/glm53/models/glm53-abliterated-nvfp4e-int8r-s0r4 \
  --work /work
```

- `--out` gets 34 new shards (same names as the shards they replace), a new `model.safetensors.index.json` and a new
  `config.json`; every other file of `--model` is hard-linked into it (`--link hardlink`, the default, needs one
  filesystem). Elsewhere use `--link copy`, or `--link symlink` and mount `--model` into the engine container at the
  same path as well.
- `--keep-int8` sets which of the layer 0-29 modules stay INT8. The default is the served choice:
  `self_attn.q_a_proj,self_attn.kv_a_proj_with_mqa,self_attn.kv_b_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj`.
- `--work` holds an amax cache (`stage0_amax.json`) and a report; a rerun reuses the cache. `--only-shards` and
  `--skip-shards` resume a partial run.
- The new `config.json` narrows `quantization_config` group_1 (INT8) to layers 30-77 plus the modules kept at INT8,
  and adds group_2 (`nvfp4-pack-quantized`) for the 4-bit modules. vLLM fuses q_a + kv_a and the shared experts'
  gate + up; a fused pair gets one global scale.

`s0r4.sha256` lists the sha256 of the 34 shards, the index and the config that this tool wrote for the served
checkpoint from the inputs above. A run with the same inputs and the same image writes the same bytes (checked for
one shard, the index and the config); with other library versions the bytes may differ.

## Serve it

Point `HOST_MODEL_DIR` in `multinode/host.env` at the new directory on both hosts and start
`multinode/layouts/pp5bw30-s0r4bs.env` ([multinode/README.md](../../multinode/README.md)). The stage-0 engine log
then reports the Marlin NVFP4 kernel for the 4-bit modules.

## What it changed

- Stage 0 reads fewer weight bytes per step: its chunk time fell from 9.05 to 8.36 ms, worth 2 to 3% single-stream
  speed, and the 2.4 GB it frees raised the KV pool from 817,024 to 873,536 tokens.
- Quality: 89.76 to 89.87% top-1 agreement over three readings (2026-11-06 and 2026-11-11), with an NLL 0.84 to 1.22%
  lower than the reference. The reference outputs were captured with stage 0 at INT8, so a 4-bit stage 0 reads as a
  small agreement drop by construction; a blind comparison of 32 matched outputs found no difference
  ([docs/quality-floor.md](../../docs/quality-floor.md)).

## Licence

The tool is MIT ([LICENSE](../../LICENSE)). It calls compressed-tensors (Apache-2.0) as a library and copies none of
its code. The checkpoints have their own licences; see their model cards.
