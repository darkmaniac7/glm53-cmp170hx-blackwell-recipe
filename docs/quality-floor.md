# The quality floor

A layout that fails the floor is not served, whatever its speed. Three checks, all against the engine's own port.

## 1. Teacher-forced agreement against TP8 goldens

- Tool: cebeuq's `bench/quality_gate.py`, unchanged.
- Goldens: `quality_gate.py capture` from the same checkpoint served at TP8, where there is no pipeline and so no
  stage hand-off. 32 prompts with their greedy continuations and token log-probs, plus the NLL of a fixed text.
- Compare: `quality_gate.py compare <url> goldens-tp8.json report.json` against the candidate layout. Read the
  top-1 agreement per position, the NLL delta, the distribution of per-token log-prob differences (mean and p99),
  non-finite values and collapsed free runs.
- Floor: **89.9% agreement**. Every correct layout of this checkpoint has measured between 90.0 and 90.4 against
  these goldens; the one broken configuration measured 87.0.

The tool prints its own verdict from a 95% tier that was calibrated for cebeuq's goldens, so with these TP8
goldens its `pass` field reads false for every layout. Read the agreement number against the floor above. One collapsed free run on a math prompt (a greedy loop) shows up now and then and
is marked acceptable by the tool's second tier.

| Configuration | Agreement | NLL vs goldens | Log-prob difference, mean / p99 |
|---|---:|---:|---:|
| 8 CMPs, TP2 x PP4, greedy drafts | 90.13% | -0.65% | 0.055 / 0.71 |
| 8 CMPs, TP2 x PP4, probabilistic drafts | 90.07% | -0.69% | 0.057 / 0.74 |
| 8 CMPs, TP2 x PP4, WITHOUT the hand-off fix | 86.98% | +0.38% | 0.164 / 1.92 |
| Two hosts, pp5bw30, probabilistic drafts | 90.18% | -0.02% | 0.056 / 0.76 |
| Two hosts, pp5bw30, greedy drafts | 90.15% | -0.10% | 0.058 / 0.74 |
| Two hosts, pp5bw30, three draft tokens | 90.19% | +0.00% | 0.056 / 0.71 |
| Two hosts, pp5bw30, without the grammar overlay | 90.38% | -0.96% | 0.056 / 0.73 |

The gate runs at temperature 0. For sampled requests the check is the NLL of sampled text under the model (6 x 384
tokens): 0.61 to 0.70 with greedy and with probabilistic drafts alike, which is the probe's own run-to-run spread.

## 2. Structured-output probe

`struct_probe.py` (in [overlays/g4](../overlays/g4/bench/struct_probe.py)) at concurrency 1 and 8: json_schema
strict, a named tool choice, json_object and an unconstrained control. Pass means no HTTP 5xx and every constrained
reply parses and matches its schema.

## 3. Speculative acceptance

Accepted draft tokens per verified chunk, from the engine's spec-decode counters, read per workload (delta of the
counters across the workload). It says whether a speed difference comes from the drafter or from the layout: the
two-host layout and the eight CMPs accept the same 1.28 to 1.30 drafts per chunk at temperature 0, so the speed
difference between them is the hardware.
