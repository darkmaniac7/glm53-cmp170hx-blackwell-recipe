# Overlays for cebeuq's recipe

Three additions to [cebeuq/glm53-cmp170hx-recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe), in its own
form: whole-file patches for the pinned image under `patches/vllm/`, a compose overlay under `engine/` that
bind-mounts them read-only, and a diff of every file against the image original under `patches/diffs/`. Each
overlay directory mirrors the recipe's tree, so applying one is a copy.

| Overlay | What it does | Compose file | Offered upstream as |
|---|---|---|---|
| [ppfix](ppfix/) | pipeline hand-off fix for TP > 1 | `compose.ppfix.yaml` | [pull request 1](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/1) |
| [g4](g4/) | grammar bitmask for draft tokens: structured output with speculative decoding | `compose.grammar.yaml` | [pull request 2](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/2) |
| [sq](sq/) | probabilistic drafts under PipeSpec for sampled requests | `compose.sq.yaml` | [pull request 3](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/3) |

They were written against the recipe at commit `31c0934958a8b0892b6de139e37d2bfe18926ae4` and the image
`lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e`. Mounted over any
other image they will break it; port with the diffs.

## Apply

```bash
git clone https://github.com/cebeuq/glm53-cmp170hx-recipe
tools/apply-overlays.sh glm53-cmp170hx-recipe            # all three; or name them: ... ppfix g4
```

Then, in the recipe's `engine/prod.vars`, extend the overlay chain. Order matters: a later file replaces an
earlier mount of the same container path.

```
COMPOSE_FILE=compose.yaml:compose.p2p.yaml:compose.kvfp8.yaml:compose.kvfp8i8.yaml:compose.pipespec.yaml:compose.ppfix.yaml:compose.grammar.yaml:compose.sq.yaml
```

and, for the probabilistic drafts, add `"draft_sample_method":"probabilistic"` to the speculative config in
`EXTRA_ARGS`:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}'
```

`docker compose --env-file prod.vars config` shows the resulting mounts; the full chain ends with the 22 files of
[multinode/mounts.list](../multinode/mounts.list).

## Combinations

- `ppfix` stands alone and belongs in every chain with TP > 1 and PP > 1.
- `g4` needs `compose.pipespec.yaml` before it (set `CMP_PIPESPEC=0` there if PipeSpec is not wanted).
- `sq` needs `compose.pipespec.yaml`. With `g4` in the chain use `compose.sq.yaml` (after `compose.grammar.yaml`);
  without `g4` use `compose.sq-nogrammar.yaml`. The two differ only in the model runner they mount. Do not turn on
  probabilistic drafts with a model runner that lacks the sq hunks: chunk B would be verified against the wrong
  draft logits and sampling would no longer be exact.

## Licence of the patch files

Everything under `patches/vllm/` is vLLM code (Apache-2.0) from the pinned image with changes, or cebeuq's patched
copy of it with further changes, and keeps its `SPDX-License-Identifier: Apache-2.0` header. The diffs show exactly
what changed against the image; the `delta/` directories show what changed against cebeuq's file where a file
derives from one of his.
