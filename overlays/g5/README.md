# g5: structured-output requests keep their drafts (replaces the g4 fix)

Structured output (`response_format` json_schema or json_object, a named `tool_choice`) together with speculative
decoding, with several streams on the pipeline. g5 is a port of vLLM PR 63219 (per-request `DraftTokensHandler`)
and replaces the [g4](../g4/) approach as the fix.

**What g4 left.** Under async scheduling and the pipeline's decode stagger, another micro-batch samples between a
request's draft step and the scheduler's fetch for that request's next step. The worker kept the latest drafts in
one slot that every sampled batch overwrote, so the fetch returned the other batch's rows and the grammar bitmask
was built from `[-1] * k` placeholders. g4 (a port of vLLM PR 54442) made that safe: drafts verified against a
placeholder row are rejected, so no reply breaks the grammar and nothing ends in HTTP 500. The price is the drafts
themselves. With eight streams the structured requests kept few of their drafts, and constrained decode ran below
free decode.

**Why it was replaced.** vLLM reverted PR 54442 for exactly that cost (a performance regression under pipeline
parallelism with speculative decoding and structured outputs) and fixed the race at its source in PR 63219, merged
2026-10-30: the worker keeps each request's drafts until the scheduler has fetched them. g5 ports that onto the
pinned image and onto the PipeSpec scheduler and model runner, so a deferred step's bitmask is built from the real,
grammar-validated drafts and the request keeps them.

**What stays from g4.** Five of g4's seven files stay mounted: `GrammarOutput.num_acceptable_drafts`, the
worker-side mask and the rejection sampler are still there, now as a net for a draft window that still holds
placeholders (sampling was not deferred, or the worker had no drafts for the request). With the drafts kept per
request the net did not fire once in the measurement below. `CMP_G5_NET=0` turns the net off and gives PR 63219 as
merged. The g4 scheduler and model runner are replaced by g5's; the g4 files stay in this repository for history
and for the five mounts.

| File | Mounted over (`vllm/...`) | Derived from |
|---|---|---|
| `g5_spec_decode_utils.py` | `v1/worker/gpu/spec_decode/utils.py` | image: `DraftTokensHandler` keeps drafts per request, `begin_step`, `discard` |
| `g5_scheduler.py` | `v1/core/sched/scheduler.py` | g4's `grammar_scheduler.py`: full draft windows count, the net, the counter |
| `g5_model_runner.py` | `v1/worker/gpu/model_runner.py` | sq's `sq_grammar_model_runner.py`: two calls into the handler, one comment |

`delta/` has each file against the repository file it derives from, `patches/diffs/` against the image original,
`sha256.txt` the checksums of the three files (`sha256sum -c sha256.txt` in this directory). The model runner
carries the sq hunks; they do nothing unless `"draft_sample_method":"probabilistic"` is set.

**Measured** 2026-11-05 on the two-host layout (TP2 x PP5, two RTX PRO 6000 Blackwell + eight CMP 170HX, MTP k=2
+ PipeSpec, probabilistic drafts, ppfix, 524K context, 8 sequences). The g4 engine and the g5 engine back to back,
same probe (`bench/struct_probe.py` of the g4 overlay: json_schema, named tool_choice, json_object and an
unconstrained control), at concurrency 8:

| | g4 | g5 |
|---|---:|---:|
| Accepted drafts per step, structured requests | 0.31 | 1.60 |
| Per-stream speed of each constrained kind | 1x | 1.75 to 1.85x |
| Aggregate of the probe, tok/s | 76.6 | 133.4 |
| Draft windows left with placeholders (g5 counter) | | 0 |

At concurrency 1 nothing changes: the drafts were not lost there. No HTTP 5xx and no grammar error on either
engine. The quality gate, single-stream speed and the long-context chat stayed inside their run-to-run spread.
One json_object reply of the g5 run came back with empty content. The g4 engine returns such replies too, at a
rate the measurement could not tell apart: it is the model ending its turn inside the thinking block, not this
overlay. [jor](../jor/) fixes it.

**Not measured.** The eight CMPs alone; `CMP_G5_NET=0`; streaming requests; concurrency above 8; hours of mixed
traffic; preemption under KV pressure; long constrained replies.

**Switches.** `CMP_G5_NET` (default 1: keep the g4 invalidation as the net; 0: PR 63219 as merged). `CMP_G5_LOG`
(default 2000): the leader logs `CMP g5: <n> structured draft windows, <m> left with placeholders (net on)` every
n windows; 0 turns it off. A healthy engine keeps m at 0.

**Use.** Append `:compose.g5.yaml` to `COMPOSE_FILE` after `compose.grammar.yaml` and `compose.sq.yaml`
([overlays/README.md](../README.md)). Check with the g4 probe:
`python3 bench/struct_probe.py http://127.0.0.1:8000 glm-5.3 8 3 c8`.
