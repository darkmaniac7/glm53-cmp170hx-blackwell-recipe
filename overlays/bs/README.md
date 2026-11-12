# bs: two PipeSpec overhead cuts

For the pipelined decode of one stream under PipeSpec (MTP with 2 draft tokens, chunks A and B in flight). Two
switches, each a small reordering of when a message is posted or sent. The messages, their order on every
communicator and the math are unchanged.

- `CMP_BS_RXLATE=1` (`bs_pp_utils.py`, with a flush in `bs_model_runner.py`): the middle pipeline ranks (ranks 1 to
  PP-2) post their broadcast receive for a chunk-A step after the forward of the chunk B behind it, not before it.
  On a CMP 170HX a pending NCCL broadcast receive slows the forward running beside it, so that forward no longer
  waits on it. A deferred receive is always posted before any later one, so the collective order on the broadcast
  communicator does not change; when no chunk B follows, the receive is posted at once. `CMP_BS_RXLATE_RANKS`
  (comma-separated ranks) overrides the default rank set.
- `CMP_BS_P1EARLY=1` (`bs_model_runner.py`): the last rank, when it drops a chunk B, sends that step's message and
  the first draft message after it (P1) from the step's `execute_model`, right after the drop decision, instead of
  one call later, and runs the second draft chain (P2) behind them. Nothing is sent before the step that proves
  chunk B exists: the scheduler can leave a chunk A without its chunk B (abort, preemption, pause).

`compose.bs.yaml` sets both switches to 1. With both at 0 the two files behave as the files they derive from.

| File | Mounted over (`vllm/...`) | Derived from |
|---|---|---|
| `bs_model_runner.py` | `v1/worker/gpu/model_runner.py` | g5's `g5_model_runner.py` (which carries the sq and g4 hunks) |
| `bs_pp_utils.py` | `v1/worker/gpu/pp_utils.py` | the recipe's `pipespec_pp_utils.py` |

`delta/` has each file against the file it derives from (added lines only), `patches/diffs/` against the image
original, `sha256.txt` the checksums (`sha256sum -c sha256.txt` in this directory). The two files are the served
ones with comment tags rewritten and one log message shortened; the code is otherwise identical.

**Measured** 2026-11-11 on the two-host layout in its current form (TP2 x PP5, stage 0 at 4-bit, MTP k=2 + PipeSpec,
probabilistic drafts, g5 and jor, 524,288-token context, 8 sequences), against the same layout without the cuts in
one window:

| | without the cuts | with the cuts |
|---|---:|---|
| Forward of chunk B on the first two CMP stages | 5.4 ms | 4.5 ms |
| Last stage, dropped step: message ready to sent | 0.81 ms | 0.02 ms |
| Single-stream epoch | 35.59 ms | 34.15 ms (4.07% shorter) |
| One-user speed, MiaAI-Lab protocol | | 4.0% higher |
| Eight users | | unchanged |

A 45-minute mixed soak on the cuts (942 requests, streams joining and leaving, client aborts in mid-decode) had 0
errors. PipeSpec runs only while one stream decodes alone, so the cuts do not touch multi-stream decode.

**Not measured.** The eight CMPs alone (TP2 x PP4); a layout with more or fewer pipeline stages; the rank override.

**Use.** Append `:compose.bs.yaml` to `COMPOSE_FILE` after `compose.g5.yaml`, and after `compose.jor.yaml` when it
is in the chain ([overlays/README.md](../README.md)). On the two-host layout: `multinode/layouts/pp5bw30-s0r4bs.env`.
