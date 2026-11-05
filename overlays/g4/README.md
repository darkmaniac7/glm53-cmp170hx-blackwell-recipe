# g4: grammar bitmask for draft tokens

Structured output (`response_format` json_schema or json_object, a named `tool_choice`) together with speculative
decoding. The full write-up is [docs/structured-output.md](docs/structured-output.md).

**What breaks.** With MTP drafts on, such requests could end in HTTP 500 (`Unexpected: grammar rejected tokens`):
4 of 24 requests of a structured-output test set with PipeSpec, 3 of 24 without it, none with speculation off
(2026-10-27/28, json_schema and named tool_choice). Under async scheduling and the pipeline's decode stagger a lone request's placeholder
drafts are consumed before it is scheduled again, so the grammar bitmask is built from `[-1] * k` placeholders: the
draft rows allow everything and the bonus row is stale. The worker can then commit a draft the grammar would not
have allowed, and the request fails.

**The fix.** A port of vLLM PR 54442 onto the pinned image and onto cebeuq's PipeSpec scheduler and model runner:
`GrammarOutput.num_acceptable_drafts`, a worker-side mask of drafts that met a permissive row, and a rejection
sampler that will not accept them. Seven files, mounted by `engine/compose.grammar.yaml` after
`compose.pipespec.yaml`:

| File | Mounted over (`vllm/...`) | Derived from |
|---|---|---|
| `grammar_sched_output.py` | `v1/core/sched/output.py` | image |
| `grammar_scheduler.py` | `v1/core/sched/scheduler.py` | cebeuq's `pipespec_scheduler.py` (14 changed lines) |
| `grammar_structured_output_init.py` | `v1/structured_output/__init__.py` | image |
| `grammar_gpu_structured_outputs.py` | `v1/worker/gpu/structured_outputs.py` | image |
| `grammar_batch_shard.py` | `v1/worker/gpu/sample/batch_shard.py` | image |
| `grammar_rejection_sampler.py` | `v1/worker/gpu/spec_decode/rejection_sampler.py` | image |
| `grammar_model_runner.py` | `v1/worker/gpu/model_runner.py` | cebeuq's `pipespec_model_runner.py` (14 changed lines) |

**Measured** 2026-11-04 on eight CMP 170HX (TP2 x PP4, MTP k=2 + PipeSpec, 524K context, 8 sequences, with ppfix):

- `bench/struct_probe.py` at concurrency 1 (10 requests) and 8 (24 requests): no 5xx, every json_schema, named
  tool_choice and json_object reply valid.
- Requests without a grammar: unchanged within noise (one stream on a code prompt 94.5 / 89.7 / 94.3 tok/s with,
  86.5 / 87.3 / 88.2 without; 17K cold prefill 2,234 against 2,210 tok/s). On the two-host layout the engine with and
  without this overlay was equal on every row.
- Cost: a constrained request whose bitmask came from placeholders loses its drafts for that step. At one stream
  the constrained replies ran at 42 tok/s (named tool_choice), 49 and 70 (two json_schema prompts) and 60
  (json_object) where an unconstrained one ran at 75.

**Limits.** vLLM later replaced this fix with one at the source of the race (PR 63219, per-request draft retention),
which keeps the drafts of constrained requests. That is not ported here.

**Use.** Append `:compose.grammar.yaml` to `COMPOSE_FILE` after `compose.pipespec.yaml`. Check with
`python3 bench/struct_probe.py http://127.0.0.1:8000 glm-5.3 1 10 c1` and `... 8 3 c8`.
