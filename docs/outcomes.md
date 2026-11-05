# Outcomes

What the two-host layout measured, with the conditions each number needs. For the method behind the quality row see
[quality-floor.md](quality-floor.md); for how much a single number moves between reloads see [noise.md](noise.md).

## Conditions

- Layout `pp5bw30` ([multinode/layouts/pp5bw30.env](../multinode/layouts/pp5bw30.env)): tensor parallel 2 x pipeline
  parallel 5. Stage 0 is a pair of RTX PRO 6000 Blackwell on one host (layers 0-29 and the embedding); stages 1-4 are
  four pairs of CMP 170HX on a second host (12 layers each, the last pair also holds the norm, lm_head and the MTP
  drafter layer). The hosts are joined by a direct 100 GbE link (ConnectX-5), NCCL over IB verbs on it.
- Checkpoint: GLM-5.3 (753B MoE), NVFP4 experts with INT8 residue (the hybrid of cebeuq's recipe, built from the
  abliterated NVFP4 checkpoint), 288 files, 418 GiB.
- Engine: cebeuq's recipe (pinned sm_80 vLLM image and patches) plus this repository's three overlays. MTP with 2
  draft tokens, PipeSpec, probabilistic drafts, P2P custom all-reduce inside each CMP pair, int8-tile KV cache.
- 524,288-token context, 8 sequences, measured at the engine's own port on the night of 2026-11-04/05.
- Single-stream rows are 1,024-token answers to the same three prompts: a JSON array, a Python module, a 900-word
  essay. "Sampled" means temperature 0.9, top_p 0.9, top_k 80, presence penalty 0.5, repetition penalty 1.02.

## The numbers

| What | Number |
|---|---|
| One stream, temperature 0: JSON / code / prose | 95 / 104 / 66 tok/s (median of 3 runs) |
| One stream, sampled: code / prose | 102 / 66 tok/s (median of 3 runs) |
| 8 concurrent streams, aggregate: JSON / code / prose | 240 / 244 / 194 tok/s (28-30 per stream) |
| Cold prefill (unique text, no prefix cache): 17K / 27K / 64K tokens | 3,400 / 3,800 / 4,100 tok/s (64K tokens in 15.5 s) |
| Long-context chat, one stream: first token at an 83K-token prompt, then a warm turn | 20 s cold, 5.7 s warm; 68-74 tok/s per stream |
| Long-context chat, four concurrent streams at 83K each | first token p50 14 s (the four cold prefills queue: 22 to 87 s), warm turns 6 s, 33 tok/s per stream, 76 aggregate |
| Three concurrent 100K-token coding sessions, three turns each | first turn 55 s median, later turns 3.2 s, 29 tok/s aggregate |
| Structured output (json_schema strict, named tool choice, json_object) at 1 and 8 streams | no HTTP 5xx, every reply valid; constrained decode 40-73 tok/s single stream |
| KV cache capacity | 817,024 tokens (1.56 x the 524,288-token maximum request) |
| Quality floor | 90.2% top-1 agreement with the TP8 goldens, NLL within 0.02% |
| Speculative acceptance (accepted drafts per verified 2-draft chunk) | 1.28 at temperature 0, 1.21 sampled on the same prompts, 1.07-1.20 on the long-context chat turns that night (0.97 to 1.61 across later reruns: it depends on the text a run writes); 3.0 committed tokens per PipeSpec epoch |

For scale: the same checkpoint on the eight CMPs alone (cebeuq's TP2 x PP4 with the same engine features, measured
on the same CMP host) runs 90-95 / 93 / 61 tok/s single stream and 187 / 216 / 151 aggregate at 8 streams (the
8-stream row with greedy drafts, before the grammar overlay), prefills a cold 17K prompt at 2,270 tok/s, reaches
the first token of an 83K prompt in 32 to 38 s cold and 8.4 s on a warm turn, and holds 569,792 KV tokens. The
Blackwell stage adds about 10% single-stream, 13 to 28% 8-stream aggregate, about 50% prefill rate and 43% KV.

## What each engine feature buys

- **MTP drafting with PipeSpec** is the single-stream speed. On this layout on 2026-11-03 (greedy drafts, hand-off
  fix on, no grammar overlay): no speculation 48 tok/s, MTP k=2 alone 81 / 82 / 65, MTP k=2 with PipeSpec
  97 / 102 / 67 (JSON / code / prose). Eight-stream aggregate does not change with PipeSpec; it acts on a lone
  request only.
- **Probabilistic drafts** (overlay `sq`) buy accepted drafts where the sampled distribution is flat. Greedy and
  probabilistic arms back to back on this layout: pooled sampled acceptance 1.059 against 1.147 (+8%), raw prose
  +12%, sampled chat prompts unchanged (1.212 against 1.207), nothing on code, at temperature 0 or at four streams.
  The long-context chat turns read +8 to +20%, but that workload alone spreads further than that between reruns
  ([noise.md](noise.md)). Every tok/s difference against greedy drafts on this layout is inside the noise.
- **The pipeline hand-off fix** (overlay `ppfix`) is about 3 points of agreement with the TP8 reference. Without it
  the eight-CMP layout fails the quality floor (86.98%); it was not switched off on the two-host layout. It costs
  no speed.
- **The grammar bitmask fix** (overlay `g4`) is what makes structured output usable with drafts on. Requests without a
  grammar measured the same with and without it.

## Choosing the split

Measured on 2026-11-03 with greedy drafts, the hand-off fix on and no grammar overlay, same hardware:

- Blackwells as the first stage beat Blackwells as the last stage: 8-stream aggregate 254 / 254 / 195 against
  246 / 243 / 167, single stream equal within 2%, and prefill is better because every CMP stage can drop to 12 layers.
- 30 layers on the Blackwell pair is the balance point at 524K context. 34 layers fit only at 262K context and add
  0 to 1% single-stream and 2 to 3% prefill.
- The Blackwells as two TP1 stages (which forces TP1 on every CMP stage too) give the best prefill (64K at
  4,676 tok/s) and lose 25 to 30% single-stream.
- Per layer the Blackwell pair computes about 2.65 times faster than a CMP pair in prefill, so 30 / 12 / 12 / 12 / 12
  is roughly compute-balanced, and about half of every stage's prefill time is pipeline wait.

## The 6+2 variant

`l62b` ([multinode/layouts/l62b.env](../multinode/layouts/l62b.env)) runs the same engine on the two Blackwells and
six CMPs and leaves two CMPs free. Measured 2026-11-05, back to back with `pp5bw30`:

| | pp5bw30 (30,12,12,12,12) | l62b (30,17,17,14) | 30,16,16,16 |
|---|---:|---:|---:|
| GiB used per card on the CMP stages | 32.5 / 32.5 / 32.5 / 43.7 | 45.3 / 45.3 / 48.5 | 42.7 / 42.7 / 53.8 |
| KV pool, tokens | 817,024 | 817,024 | 606,464 |
| Load, stop to ready | 377-390 s | 396 s | 397 s |
| One stream, temperature 0: JSON / code / prose, tok/s | 98.3 / 100.5 / 66.2 | 96.4 / 100.5 / 65.3 | 99.6 / 103.3 / 65.6 |
| One stream, sampled: code / prose, tok/s (mean of 3) | 103.6 / 67.1 | 101.3 / 66.6 | 101.9 / 66.2 |
| Cold prefill, 83K tokens, tok/s (two runs) | 4,144 / 4,164 | 3,240 / 3,131 | 3,274 / 3,303 |
| Cold prefill, 64K tokens, tok/s (two runs) | 4,141 / 4,157 | 3,275 / 3,286 | 3,306 / 3,315 |
| Long-context chat, one stream: first token cold / warm | 20.1 / 5.6 s | 25.7 / 6.3 s | 25.3 / 6.5 s |
| Four long chats started 20 s apart, three turns each: first token p95 | 25.9 s | 50.8 s | 42.1 s |
| The same mix: cold-start p50 / warm-turn p95 | 23.4 / 14.3 s | 40.5 / 20.5 s | 36.0 / 20.2 s |
| The same mix: aggregate tok/s | 71.0 | 56.6 | 58.6 |

- Decode does not slow down: 78 layers either way, one hop fewer.
- Prefill is set by the largest CMP stage: 12 layers 4.15K tok/s, 16 layers 3.29K, 17 layers 3.19K. That is 21 to 23%
  slower, so every cold 83K start costs 5.5 s more and every warm turn 0.8 s more, and with four chats the cold
  starts queue.
- Use 30,17,17,14 rather than 30,16,16,16: the 14-layer last stage (which also carries lm_head, sampling and the
  drafter) keeps the KV pool bound by the Blackwell stage. The two 6+2 columns differ by one sample each in the
  four-chat mix.

## What did not help

On `pp5bw30` unless a line says otherwise, 2026-11-04/05, as outcomes:

- **Three draft tokens.** PipeSpec handles exactly 2, so k=3 runs as plain MTP: 2.77 committed tokens per verify step
  against 3.0 to 3.1 per PipeSpec epoch, 9 to 13% slower single-stream.
- **A third PipeSpec chunk.** The third chunk was valid in 0.20 of epochs on code and 0.03 to 0.08 on prose and chat,
  the second chunk's valid rate fell by about as much as the third added, each Blackwell drew 75 W more, and tok/s
  stayed where it was.
- **Scheduler flags for the first-token tail.** With several long chats the tail is the cold prefill itself (83K
  tokens = 20 s) plus first-come-first-served queueing. `--long-prefill-token-threshold 1024` made the four-chat p95
  23% worse (it helps warm turns only when two chats play); `--max-num-batched-tokens 4096` with a 2048 threshold
  gave no extra prefill rate, 4% less KV and twice the worst stall; `CMP_FAIR_CHUNK=1024` cut in-stream stalls by 40%
  and delayed first tokens. No flag of this engine lowers that tail.
- **`CMP_PIPESPEC_REFILL=1`.** No change in acceptance or speed.
- **Block verification** (`rejection_sample_method: block`), measured on the eight CMPs. Exact, and no gain at k=2
  with these acceptance rates, with greedy or with probabilistic drafts.
- **A greedy bonus guess under probabilistic drafts** (`CMP_SQ_GREEDY_D3=1`), measured on the eight CMPs. Exact, no
  gain.
- **Truncating the draft proposal to the request's top_k / top_p.** Exact, +1.4% pooled acceptance against a noise
  floor of 4%: not resolvable, not adopted.
