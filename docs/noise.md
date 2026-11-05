# Noise: how far one number moves

Greedy output of this engine is not reproducible run to run (cebeuq's recipe documents this; it predates every patch
here). The text a run writes changes which drafts are accepted, so the same configuration reloaded gives different
speeds. Before comparing two arms, know the spread of one.

Same day (2026-11-05), layout `pp5bw30`, seven to eight reloads or reruns of configurations that differ only in
scheduler flags or in switches that leave single-stream output untouched:

| Metric | Samples | Spread |
|---|---|---|
| One stream, temperature 0, code prompt, tok/s | 100.5, 102.7, 106.3, 100.5, 99.1, 98.9, 98.7 | 98.7 to 106.3 (about +-4%) |
| One stream, temperature 0, prose prompt | 66.2, 63.6, 65.2, 67.4, 65.3, 67.9, 65.1 | 63.6 to 67.9 (about +-3%) |
| One stream, temperature 0, JSON prompt | 98.3, 94.2, 92.7, 94.5, 92.9, 99.1, 96.0 | 92.7 to 99.1 |
| One stream, sampled prose, mean of 3 | 67.1, 67.6, 67.1, 67.7, 67.5, 66.3, 67.7, 65.7 | 65.7 to 67.7 (about +-1.5%): the tightest speed number |
| Four staggered long chats, first token p95, s | 25.9, 25.2, 26.1 | about +-2% |
| Cold prefill of 83K tokens, tok/s | 4,141 to 4,164 | about +-0.3% |
| Long-context chat, one stream, first token p50 | 12.7 to 12.9 s on every arm | about +-1% |
| Long-context chat, one stream, tok/s per stream | 62.9, 70.6, 83.6, 64.2, 73.9, 72.5, 98.5 | 62.9 to 98.5: it depends on the text the run writes (0.97 to 1.61 accepted drafts per chunk) and cannot rank arms |
| NLL of sampled text (6 x 384 tokens) | 0.613 (the night before), 0.699, 0.665, 0.663 | 0.61 to 0.70 |

On the eight CMPs alone (2026-11-04) three arms with identical temperature-0 math read 1.303, 1.305 and 1.251 accepted
drafts per chunk and 94.7, 89.6 and 85.6 tok/s on the JSON prompt, with clocks and temperatures flat.

What follows from it:

- A difference of 5% or less between two single-stream tok/s numbers is not a result. Prefill rates and first-token times at a
  fixed prompt are tight (1% or better) and can be compared directly.
- Acceptance has a counting error of about 0.015 on 2,800 chunks and a text-divergence spread of about 4%. Pool
  several workloads, or interleave the arms (A, B, A, B) and compare the pooled counters.
- Eight-stream aggregate from single runs spreads widely: three runs on one engine read 222, 240 and 241 tok/s on
  the JSON prompt, 219, 244 and 246 on code, and 194, 172 and 282 on prose. Run it three times and quote the median.
- The first long prompt after a load is slow once (27 s against 20 s for the same 83K prompt two minutes later).
  Measure first-token times on a warmed engine.
- Cumulative PipeSpec counters from engines with different histories are not comparable. Compare intervals.
