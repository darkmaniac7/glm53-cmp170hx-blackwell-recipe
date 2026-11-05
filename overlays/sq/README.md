# sq: probabilistic drafts under PipeSpec

**What it is.** PipeSpec required greedy (argmax) drafts. For a sampled request the verifier then treats each
draft as a one-hot proposal, which is exact but wastes drafts where the sampled distribution is flat. With this
overlay PipeSpec accepts `"draft_sample_method":"probabilistic"`: the drafter samples at the request temperature
(Gumbel) and both PipeSpec chunks are verified against the distribution that produced their drafts. The committed
tokens are distributed exactly as without speculation.

**Files.** Three of cebeuq's PipeSpec files with changes (8, 40 and 20 changed lines; `delta/` has the diffs
against his files):

| File | Mounted over (`vllm/...`) | Change |
|---|---|---|
| `sq_pipespec_core.py` | `v1/cmp_pipespec.py` | the gate accepts probabilistic drafts |
| `sq_pipespec_ar_speculator.py` | `v1/worker/gpu/spec_decode/autoregressive/speculator.py` | a draft-logits cache of k + (k + 1) = 5 columns, so the chain's steps keep their logits |
| `sq_grammar_model_runner.py` | `v1/worker/gpu/model_runner.py` | g4's model runner + chunk B is verified with its own draft logits; restored drafts get one-hot logits |
| `sq_pipespec_model_runner.py` | `v1/worker/gpu/model_runner.py` | the same two hunks on cebeuq's `pipespec_model_runner.py`, for a chain without g4 |

`engine/compose.sq.yaml` mounts the first three (after `compose.grammar.yaml`); `engine/compose.sq-nogrammar.yaml`
mounts the first two and the fourth. The three files of the upstream pull request are the first two and the fourth
under cebeuq's own file names.

**Exactness.** `pipespec/check_sampling.py` (CPU, standard library) runs the chunk A/B protocol on a toy model with
greedy and sampled drafts, standard and block verification, and compares the frequency of every committed sequence
with its target probability: `python3 pipespec/check_sampling.py` prints `RESULT ALL EXACT`. On hardware: the quality
gate at temperature 0 and the NLL of sampled text are unchanged.

**Measured** 2026-11-04 on eight CMP 170HX (TP2 x PP4, MTP k=2 + PipeSpec, 524K context, with ppfix and g4), greedy
against probabilistic drafts; "sampled" is temperature 0.9, top_p 0.9, top_k 80, presence penalty 0.5, repetition
penalty 1.02; acceptance is accepted drafts per verified 2-draft chunk:

| | greedy | probabilistic |
|---|---:|---:|
| Acceptance, sampled chat prompts (6 x 1,024 tokens) | 1.191 | 1.252 |
| Acceptance, sampled long-context chat (83K prompt, 2 turns x 2,000 tokens) | 1.226 | 1.385 |
| Acceptance, sampled raw prose completions (6 x 384 tokens) | 0.829 | 0.929 |
| Acceptance, pooled | 1.124 | 1.223 (+8.8%) |
| One stream, sampled, code / prose tok/s (median of 3) | 92.0 / 59.4 | 94.6 / 61.8 |
| Raw prose completions, tok/s | 50.0 | 56.9 |
| Long-context chat, mean of two turns, tok/s (one run) | 68.4 | 77.2 |
| Temperature 0: acceptance / code tok/s | 1.303 / 93.1 | 1.305 / 93.9 |
| Quality gate agreement | 90.13% | 90.07% |

On the two-host layout (greedy and probabilistic arms back to back, 2026-11-04): pooled sampled acceptance 1.059
against 1.147 (+8%), raw prose +12% accepted drafts, sampled chat prompts unchanged (1.212 against 1.207), nothing
on code, nothing at four streams. The long-context chat turns read +8 to +20%, but that workload alone spreads
further than that between reruns of one configuration. Every tok/s difference there is inside the noise; raw prose
read 59.3 tok/s with greedy and 57.5 with probabilistic drafts.

So: about 8 to 9% more accepted drafts on sampled requests, pooled, in two separate comparisons, coming from
prose-like text; nothing at temperature 0 or on peaked code output; and no throughput gain that either comparison
can separate from the +-5% run-to-run spread of one single-stream number ([docs/noise.md](../../docs/noise.md)).
The acceptance rows are the evidence, the tok/s rows are context.

**Switches.** `"draft_sample_method":"probabilistic"` in the speculative config turns it on; without it the files
behave like cebeuq's. `CMP_SQ_GREEDY_D3=1` draws the chain's bonus guess greedily; it is exact and measured no gain
(pooled acceptance 1.130), so it stays off.
