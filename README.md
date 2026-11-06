# GLM-5.3 (753B MoE) on two RTX PRO 6000 Blackwell plus eight CMP 170HX

Additions to [cebeuq's GLM-5.3 recipe for eight CMP 170HX](https://github.com/cebeuq/glm53-cmp170hx-recipe): five
overlays for his engine, and a recipe for running that engine as one pipeline across two machines, with a pair of
Blackwell cards as the first stage and the CMPs behind it.

This is not a fork and not a standalone recipe. The engine, the patches that make GLM-5.3 run on these cards, the
checkpoint conversion and the P2P driver build are cebeuq's; start there. What is here sits on top of his
repository at commit `31c0934`; the first three overlays are offered back to it as pull requests
([1](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/1), [2](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/2), [3](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/3)).

Everything below was measured on my own hardware between 2026-10-27 and 2026-11-05. Numbers carry their
conditions; where something was not measured it says so.

## In one table

| | Eight CMP 170HX (cebeuq's TP2 x PP4 + the overlays) | Two Blackwells + eight CMPs (TP2 x PP5) |
|---|---:|---:|
| One stream, temperature 0: JSON / code / prose, tok/s | 90-95 / 93 / 61 | 95 / 104 / 66 |
| 8 concurrent streams, aggregate: JSON / code / prose, tok/s | 187 / 216 / 151 (greedy drafts, before the grammar overlay) | 240 / 244 / 194 |
| Cold prefill, tok/s | 2,270 at 17K tokens | 3,400 / 3,800 / 4,100 at 17K / 27K / 64K |
| First token at an 83K-token prompt | 32 to 38 s cold, 8.4 s on a warm turn | 20 s cold, 5.7 s on a warm turn |
| KV cache capacity, tokens | 569,792 | 817,024 |
| Context | 524,288 | 524,288 |
| Quality gate, agreement with TP8 goldens | 90.1% | 90.2% |
| Structured output with drafts on, probe at 1 and 8 streams | valid, no 5xx | valid, no 5xx |
| Structured requests at 8 streams, accepted drafts per step: g4 / g5 | not measured with g5 | 0.31 / 1.60 |
| Structured requests at 8 streams, aggregate tok/s: g4 / g5 | not measured with g5 | 76.6 / 133.4 (each constrained stream 1.75 to 1.85x) |
| json_object with thinking on, 8 streams: replies with empty content, without / with the jor rule | not measured | 29 of 2,000 / 0 of 2,000 |
| Reload from local copies | 8 to 10 min warm, 22 min cold | 6 to 7 min warm, 11 min first load |

Single-stream rows are 1,024-token answers to three prompts (a JSON array, a Python module, a 900-word essay),
MTP with 2 draft tokens and PipeSpec on, 8 sequences. The structured rows with g5 and jor are from 2026-11-05: the
g4 overlay's probe at concurrency 8 with the g4 and the g5 engine back to back, and json_object on one fixed prompt
at temperature 0 and at the engine's sampling defaults. Full table, conditions and the six-CMP variant:
[docs/outcomes.md](docs/outcomes.md). One single-stream number on this engine moves by about +-5% between reloads:
[docs/noise.md](docs/noise.md).

## The overlays

Each is a set of whole-file patches for the recipe's pinned image plus a compose overlay, in the recipe's own
form, with a diff of every file against the image original.

| Overlay | What it does | Measured | Status |
|---|---|---|---|
| [ppfix](overlays/ppfix/) | Fixes the pipeline hand-off for TP > 1: a stage's un-reduced `hidden_states` were sliced per rank and all-gathered, mixing two partial sums. Settles the open question in the recipe's known limits. | Quality gate 86.98% to 90.13% agreement, NLL +0.38% to -0.65%, speed unchanged (8 CMPs, 2026-11-04) | in use since 2026-10-27 |
| [g4](overlays/g4/) | Structured output with speculative decoding: json_schema and named tool_choice requests could end in HTTP 500 when drafts were on (4 of 24 in one test set). A port of vLLM PR 54442 onto the recipe's PipeSpec scheduler and model runner. | No 5xx and every reply valid at 1 and 8 streams; unconstrained throughput unchanged; constrained decode at 55 to 92% of free decode (8 CMPs, 2026-11-04) | superseded by g5 on 2026-11-05; five of its seven files stay in use under g5 |
| [sq](overlays/sq/) | Probabilistic drafts under PipeSpec: for sampled requests the drafter samples at the request temperature and both chunks are verified against the real draft distribution. Exact by construction. | Pooled sampled acceptance +8.8% on 8 CMPs and +8% on the two-host layout; raw prose +12% accepted drafts; nothing on code or at temperature 0; tok/s differences inside the noise; gate unchanged (2026-11-04) | in use since 2026-11-04 |
| [g5](overlays/g5/) | Structured-output requests keep their drafts when several streams share the pipeline: a port of vLLM PR 63219 (drafts kept per request until the scheduler has fetched them). Replaces the g4 fix, whose vLLM PR was reverted in 63219; g4's invalidation stays only as a net. | At 8 streams: accepted drafts per step on structured requests 0.31 to 1.60, each constrained stream 1.75 to 1.85x, aggregate 76.6 to 133.4 tok/s, 0 draft windows left with placeholders, no 5xx (two-host layout, 2026-11-05) | in use since 2026-11-05 |
| [jor](overlays/jor/) | With thinking on, a request with a grammar could write its JSON inside the thinking block and end the turn there: HTTP 200, empty content. While such a request is in reasoning its stop tokens are masked and their probability goes to the closing think tag. | json_object at 8 streams on one prompt: 29 of 2,000 invalid without the rule, 0 of 2,000 with it (Fisher p = 3.4e-9); with the rule on every structured request left its thinking block, 3,080 of 3,080 (two-host layout, 2026-11-05) | optional; off unless `compose.jor.yaml` is in the chain |

How they stack on the recipe's compose files, and which combinations are valid:
[overlays/README.md](overlays/README.md).

```bash
git clone https://github.com/cebeuq/glm53-cmp170hx-recipe
tools/apply-overlays.sh glm53-cmp170hx-recipe
# then extend COMPOSE_FILE in engine/prod.vars (overlays/README.md)
```

## The two-host layout

[multinode/](multinode/) is the recipe: one documented `docker run` per node, the layout files, the mount list,
the environment, the image build for sm_120, and the procedure from weights to quality gate.

- **pp5bw30**: TP2 x PP5. The Blackwell pair holds layers 0-29 and the embedding, four CMP pairs hold 12 layers
  each. Against the eight CMPs alone it adds about 10% single-stream, 13 to 28% more 8-stream aggregate, about 50%
  faster prefill and 43% more KV.
- **l62b**: TP2 x PP4 with six CMPs (30,17,17,14), two CMPs free. Decode is unchanged, the KV pool is unchanged,
  prefill is 21 to 23% slower because the largest CMP stage sets the prefill rate.

What did not help, as outcomes ([docs/outcomes.md](docs/outcomes.md#what-did-not-help)): three draft tokens and a
third PipeSpec chunk do not pay on this layout, and no scheduler flag lowers the first-token tail with several long
chats; that tail is the cold prefill itself.

## Documents

- [Outcomes](docs/outcomes.md): every number with its conditions, the six-CMP variant, what did not help
- [Load times and storage rules](docs/load-and-storage.md): local copies, sha256, the 450 MB/s NVMe read cap and why
- [The quality floor](docs/quality-floor.md): TP8 goldens, the 89.9% floor, the structured-output probe
- [Noise](docs/noise.md): how far one number moves between reloads, and what that allows you to conclude
- [Overlays](overlays/README.md) and [the two-host procedure](multinode/README.md)

The wiki pages this repository backs:
[GLM 5.3 on 8x 8 GB plus 2x RTX PRO 6000 (vLLM TP2 x PP5)](https://cmp.wiki/wiki/GLM_5.3_on_8x_8_GB_plus_2x_RTX_PRO_6000_(vLLM_TP2_x_PP5))
and [GLM 5.3 on 8x 8 GB (cebeuq glm53-cmp170hx-recipe)](https://cmp.wiki/wiki/GLM_5.3_on_8x_8_GB_(cebeuq_glm53-cmp170hx-recipe)).

## Limits

- One pair of machines. Other boards, links and card mixes may behave differently.
- The patches are whole-file copies for one pinned image; port them with the diffs.
- The multi-node launcher in this repository is the measured one with site paths removed. Its dry run was checked
  against the original; it has not itself started an engine.
- No model weights and no driver source are in this repository.
- The checkpoint is abliterated; add the safeguards your use needs. The API has no authentication.

## Credits

- **[cebeuq](https://github.com/cebeuq)** for the [GLM-5.3 CMP 170HX recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe):
  the engine configuration, the MTP fix, the int8-tile KV cache, PipeSpec and every patch this repository builds on.
- **[Morrowmake](https://github.com/Morrowmake)** for the [GLM-5.3-Flash recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe),
  the TRAP31 P2P driver patch and the [cmpunlocker fork](https://github.com/Morrowmake/cmpunlocker).
- **lazymio** ([wtdcode](https://github.com/wtdcode)) for [vllm-backport](https://github.com/wtdcode/vllm-backport)
  and the sm_80 image everything runs on.
- **[dealignai](https://huggingface.co/dealignai)** for the abliterated NVFP4 checkpoint, and
  **[Z.ai](https://huggingface.co/zai-org)** for GLM-5.3.
- **[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)** for the memory unlock.
- The **[vLLM](https://github.com/vllm-project/vllm)** project, and the authors of its pull requests 54442 and
  63219, which the g4 and g5 overlays port.

## Licences

The documents, scripts and compose files of this repository are MIT, Copyright (c) 2026 darkmaniac7; see
[LICENSE](LICENSE). Everything under an overlay's `patches/` directory is vLLM code, or cebeuq's patched copy of
it, with changes, and keeps the Apache-2.0 SPDX header each file carries. cebeuq's recipe is MIT, Copyright (c)
2026 cebeuq. The engine image, the driver patches and the model have their own licences; see the recipe's README.
