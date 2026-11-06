# jor: no end of turn inside the thinking block (optional)

For requests with a grammar (`response_format` json_object or json_schema, a named `tool_choice`) with thinking on.
Optional: nothing changes until `compose.jor.yaml` is in the chain.

**What breaks.** Now and then the model writes the JSON inside its thinking block and then ends the turn without
closing the block. The grammar applies only after reasoning has ended, so nothing stops it, and the reply is HTTP
200 with empty content (the JSON sits in the reasoning field). The tokens of such replies are the model's own first
choices; no mask forced them. It happens with the g4 and the g5 files alike, mostly on json_object.

**The fix.** While a request with a grammar is still in its thinking block, it may not end the turn there.

- `CMP_JO_NOSTOP=1` (`jor_structured_output_init.py`, the structured-output manager): every bitmask row such a
  request gets while it is in reasoning (draft rows, rows behind a placeholder, the bonus row) loses the request's
  stop tokens (the primary EOS and its `stop_token_ids`). Everything else in those rows stays allowed, grammar rows
  are never touched, and a drafted stop token is rejected against its row like any other masked token. The model
  has to close the block before it can stop; then the grammar takes over as before.
- `CMP_JO_REDIRECT=<token id>` (`jor_gpu_structured_outputs.py`, the worker): at those positions the probability of
  the stop tokens is moved onto the closing think tag before the mask removes them
  (`logit[close] = logaddexp(logit[close], logsumexp(logit[stops]))`); every other token keeps its probability.
  Without it the removed probability spreads over the whole distribution, and at a sampling temperature a reply
  can pick a stray token before it closes the block. `154842` is `</think>` in the GLM-5.3 tokenizer; another model
  needs its own id. Needs `CMP_JO_NOSTOP=1`.
- `CMP_JO_LOG=<n>`: the manager logs a counter line every n reasoning steps of structured requests:
  `CMP jo: nostop on: ... reasoning steps of structured requests, ... rows without stop tokens, ... drafted stop
  tokens inside reasoning, ... requests ended inside reasoning`. With the switch on the last count stays 0; a
  `CMP jo: UNEXPECTED` line would be a bug.

"jor" is the two files with both switches on, which is what `compose.jor.yaml` sets by default. With both switches
at 0 the two files build the same rows as the g4 files they derive from.

| File | Mounted over (`vllm/...`) | Derived from |
|---|---|---|
| `jor_structured_output_init.py` | `v1/structured_output/__init__.py` | g4's `grammar_structured_output_init.py` |
| `jor_gpu_structured_outputs.py` | `v1/worker/gpu/structured_outputs.py` | g4's `grammar_gpu_structured_outputs.py` |

`delta/` has each file against the g4 file, `patches/diffs/` against the image original, `sha256.txt` the
checksums (`sha256sum -c sha256.txt` in this directory).

**Measured** 2026-11-05 on the two-host layout with g5 (TP2 x PP5, MTP k=2 + PipeSpec, probabilistic drafts,
ppfix, 524K context, 8 sequences), json_object with thinking on at concurrency 8 on one fixed prompt, at
temperature 0 and at the engine's sampling defaults:

| | invalid replies (empty content) |
|---|---:|
| Switches off (the engine before and after the change, and the same files with both switches unset) | 35 of 2,400 |
| `CMP_JO_NOSTOP=1` (half of the requests with `CMP_JO_REDIRECT` on as well) | 0 of 2,000 |

Fisher exact test: p = 6e-10. Over every structured request with thinking on that the engine served with the rule
on, all kinds together, the model left its thinking block 3,080 times in 3,080. No HTTP 5xx, no grammar error, no
reply cut by `max_tokens`. The schema and tool kinds stayed valid. A reply that had written its JSON inside the
thinking block now closes the block and writes it again as content, so that reply is longer; every other request
runs as before, at the same speed and with the same accepted drafts. With the mask alone, at a sampling
temperature, some of those replies left a stray token or a line of filler inside the block before closing it; with
the redirect the closing tag is the clear next token, which is why jor sets both.

**On the live engine** (two-host layout, both switches on, 2026-11-06): a 30-request json_object probe with thinking
on returned 30 valid replies, 0 empty.

**Not measured.** The eight CMPs alone; streaming requests; structured requests mixed with long chat streams in one
batch; a client stop string; `tool_choice` required or auto; a thinking block that never closes (the request would
then end by `max_tokens`); long constrained replies; soak.

**Without an engine change.** A client can skip the thinking for structured requests:
`"thinking_token_budget": 0` or `"chat_template_kwargs": {"enable_thinking": false}`. Both avoided the defect in
the same probe, at the cost of the reasoning.

**Use.** Append `:compose.jor.yaml` to `COMPOSE_FILE` after `compose.g5.yaml` ([overlays/README.md](../README.md)).
On the two-host layout: `multinode/layouts/pp5bw30-jor.env`.
