# Overlays for cebeuq's recipe

Additions to [cebeuq/glm53-cmp170hx-recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe), in its own form:
whole-file patches for the pinned image under `patches/vllm/`, a compose overlay under `engine/` that bind-mounts
them read-only, and a diff of every file against the image original under `patches/diffs/`. Each overlay directory
mirrors the recipe's tree, so applying one is a copy.

| Overlay | What it does | Compose file | Status | Offered upstream as |
|---|---|---|---|---|
| [ppfix](ppfix/) | pipeline hand-off fix for TP > 1 | `compose.ppfix.yaml` | in use | [pull request 1](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/1) |
| [g4](g4/) | grammar bitmask for draft tokens: structured output with speculative decoding | `compose.grammar.yaml` | five of its seven files in use under g5; its fix is superseded by g5 | [pull request 2](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/2) |
| [sq](sq/) | probabilistic drafts under PipeSpec for sampled requests | `compose.sq.yaml` | in use | [pull request 3](https://github.com/cebeuq/glm53-cmp170hx-recipe/pull/3) |
| [g5](g5/) | structured-output requests keep their drafts (port of vLLM PR 63219); replaces the g4 fix | `compose.g5.yaml` | in use since 2026-11-05 | |
| [jor](jor/) | no end of turn inside the thinking block for requests with a grammar | `compose.jor.yaml` | optional; in use since 2026-11-06 | |
| [bs](bs/) | two PipeSpec overhead cuts: a later receive on the middle stages, an earlier send on the last | `compose.bs.yaml` | in use since 2026-11-11 | |

They were written against the recipe at commit `31c0934958a8b0892b6de139e37d2bfe18926ae4` and the image
`lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e`. Mounted over any
other image they will break it; port with the diffs.

## Apply

```bash
git clone https://github.com/cebeuq/glm53-cmp170hx-recipe
tools/apply-overlays.sh glm53-cmp170hx-recipe            # all six; or name them: ... ppfix g4 sq g5
```

Then, in the recipe's `engine/prod.vars`, extend the overlay chain. Order matters: a later file replaces an
earlier mount of the same container path.

```
COMPOSE_FILE=compose.yaml:compose.p2p.yaml:compose.kvfp8.yaml:compose.kvfp8i8.yaml:compose.pipespec.yaml:compose.ppfix.yaml:compose.grammar.yaml:compose.sq.yaml:compose.g5.yaml
```

and, for the probabilistic drafts, add `"draft_sample_method":"probabilistic"` to the speculative config in
`EXTRA_ARGS`:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}'
```

For the optional thinking-block fix, append `:compose.jor.yaml` after `compose.g5.yaml`; its switches are on by
default in that file ([jor/README.md](jor/README.md)). For the two PipeSpec overhead cuts, append `:compose.bs.yaml`
last; its switches are on by default too ([bs/README.md](bs/README.md)). The patch set of the current form is the
chain with both:

```
COMPOSE_FILE=compose.yaml:compose.p2p.yaml:compose.kvfp8.yaml:compose.kvfp8i8.yaml:compose.pipespec.yaml:compose.ppfix.yaml:compose.grammar.yaml:compose.sq.yaml:compose.g5.yaml:compose.jor.yaml:compose.bs.yaml
```

`docker compose --env-file prod.vars config` shows the resulting mounts. The chain above ends with the 23 files of
[multinode/mounts.list](../multinode/mounts.list), the chain with jor with those of
[multinode/mounts-jor.list](../multinode/mounts-jor.list), the chain with jor and bs with those of
[multinode/mounts-bs.list](../multinode/mounts-bs.list), and the chain without g5 (the state before 2026-11-05)
with the 22 files of [multinode/mounts-g4.list](../multinode/mounts-g4.list).

## Combinations

- `ppfix` stands alone and belongs in every chain with TP > 1 and PP > 1.
- `g4` needs `compose.pipespec.yaml` before it (set `CMP_PIPESPEC=0` there if PipeSpec is not wanted).
- `sq` needs `compose.pipespec.yaml`. With `g4` in the chain use `compose.sq.yaml` (after `compose.grammar.yaml`);
  without `g4` use `compose.sq-nogrammar.yaml`. The two differ only in the model runner they mount. Do not turn on
  probabilistic drafts with a model runner that lacks the sq hunks: chunk B would be verified against the wrong
  draft logits and sampling would no longer be exact.
- `g5` needs `compose.grammar.yaml` and comes after `compose.sq.yaml`: it keeps five of g4's files, replaces g4's
  scheduler and the sq model runner with its own (which carries the sq hunks) and adds one file. It was measured
  only in the full chain above.
- `jor` comes after `compose.g5.yaml` and replaces two of g4's files. It was measured only on top of g5.
- `bs` comes after `compose.g5.yaml` (and after `compose.jor.yaml`): it replaces g5's model runner with its own,
  which carries the g5, sq and g4 hunks, and the recipe's `pp_utils.py`. It was measured only in the full chain
  above, on the two-host layout with stage 0 at 4-bit ([tools/stage0-4bit](../tools/stage0-4bit/)).

## Licence of the patch files

Everything under `patches/vllm/` is vLLM code (Apache-2.0) from the pinned image with changes, or cebeuq's patched
copy of it with further changes, and keeps its `SPDX-License-Identifier: Apache-2.0` header. The diffs show exactly
what changed against the image; the `delta/` directories show what changed against cebeuq's file where a file
derives from one of his, and, for g5, jor and bs, against the file of this repository it derives from. The
repository's [NOTICE](../NOTICE) lists what derives from what; the licence texts are in [LICENSES/](../LICENSES/).
