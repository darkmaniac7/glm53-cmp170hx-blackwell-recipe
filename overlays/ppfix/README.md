# ppfix: the pipeline hand-off at TP > 1

**What breaks.** The fork returns a non-last stage's `hidden_states` un-reduced: with `reduce_results=False` each
tensor-parallel rank holds its own partial sum of the MoE output, and the all-reduce is fused into the next layer's
input norm. The GPU worker hands `all_gather_tensors = {}` to the pipeline send and receive, so every rank sends
only its slice and the next stage all-gathers the slices. For the residual, identical on all ranks, that is right.
For the partial sums it reassembles [rank 0's first half; rank 1's second half], and the next layer's all-reduce
doubles that instead of adding the ranks. cebeuq's recipe lists this as an open question found by reading the
source; this is the confirmation and the fix.

**The fix.** `patches/vllm/ppfix_gpu_worker.py` (mounted over `vllm/v1/worker/gpu_worker.py`) sets
`"hidden_states": False` in the two places where the worker builds `all_gather_tensors`: every rank sends its whole
`hidden_states` to its peer on the next stage. Two code lines; `patches/diffs/ppfix_gpu_worker.diff`.

**Measured** 2026-11-04 on eight CMP 170HX, cebeuq's production stack (TP2 x PP4, P2P, int8 KV, MTP k=2 + PipeSpec,
524K context) with the g4 overlay on in both runs, same build and weights, with and without this overlay. Gate:
`bench/quality_gate.py compare` against goldens captured from the same checkpoint at TP8.

| | without | with |
|---|---:|---:|
| Teacher-forced top-1 agreement | 86.98% | 90.13% |
| NLL on the fixed text, against the goldens | +0.38% | -0.65% |
| Log-prob difference per token, mean / p99 | 0.164 / 1.92 | 0.055 / 0.71 |
| Accepted MTP drafts per 2-draft chunk (temperature 0) | 1.21 | 1.30 |
| One stream, JSON / code / prose prompt, tok/s | 83.3 / 94.9 / 57.8 | 94.7 / 93.1 / 60.7 |

No speed change can be resolved: code and prose are equal within the run-to-run spread
([docs/noise.md](../../docs/noise.md)), and the JSON prompt's 83.3 sits just under the 85.6 to 94.7 that reloads of
the fixed build gave with identical math. Draft acceptance is lower without the fix (1.21 against 1.25 to 1.31
across fixed reloads), which fits corrupted hidden states also hurting the drafter. Without the fix the layout fails
the 89.9% floor ([docs/quality-floor.md](../../docs/quality-floor.md)).

**Not measured.** Other TP sizes, the no-P2P variant, and the sequence-parallel branch (`enable_sp`) in which the
second changed line sits; this recipe's configuration does not take that branch.

**Use.** Append `:compose.ppfix.yaml` to `COMPOSE_FILE`. In use here since 2026-10-27, on the eight CMPs and in
every two-host layout.
