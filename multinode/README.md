# Two hosts: a Blackwell stage in front of the CMP 170HX stages

A recipe for running cebeuq's GLM-5.3 engine as one pipeline across two machines: a pair of RTX PRO 6000 Blackwell
cards holds the first 30 layers, and eight (or six) CMP 170HX hold the rest. Numbers: [docs/outcomes.md](../docs/outcomes.md).

This directory is a template, not a product. `node.sh` is the launcher this layout runs with, reduced to one
documented `docker run` per node and with every site-specific value moved into two small files. Read it before you
run it.

## What you need

- Host B: the eight-CMP machine of [cebeuq's recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe), working on
  its own first (64 GiB unlock, TRAP31 P2P driver, the hybrid checkpoint, the recipe's quality gate passing).
- Host A: two RTX PRO 6000 Blackwell (96 GB each) with working peer-to-peer between them, Docker and the NVIDIA
  Container Toolkit.
- A direct link between the hosts with RDMA NICs. Here: ConnectX-5, 100 GbE, one cable, no switch. NCCL runs over
  IB verbs on it (`/dev/infiniband` is passed into every container).
- A local copy of the checkpoint on each host ([docs/load-and-storage.md](../docs/load-and-storage.md)).

## Files

| File | What it is |
|---|---|
| `node.sh` | starts one node: builds the node's view of the weights, then one `docker run` |
| `layouts/pp5bw30.env` | TP2 x PP5: two Blackwells + eight CMPs (the layout of the outcomes page) |
| `layouts/pp5bw30-jor.env` | pp5bw30 with the optional jor overlay and its switches ([overlays/jor](../overlays/jor/)) |
| `layouts/l62b.env` | TP2 x PP4: two Blackwells + six CMPs, two CMPs free |
| `host.env.example` | per-host values: addresses, image, paths, memory, read cap |
| `mounts.list` | the 23 patch files every node mounts, with their targets (ppfix, g4, sq, g5; the default) |
| `mounts-jor.list` | the same with the two jor files in place of two g4 files |
| `mounts-g4.list` | the 22 files of the set before 2026-11-05 (ppfix, g4, sq), for history |
| `node.env` | engine and NCCL environment of every node |
| `view-build.py` | per-node directory with only the node's shards and a filtered index |
| `image/` | Dockerfiles for the two images and the sm_120 rebuild script |

## Procedure

1. **Recipe and overlays, on both hosts, same content.** Clone cebeuq's recipe and copy this repository's five
   overlays into it:
   `tools/apply-overlays.sh /path/to/glm53-cmp170hx-recipe` (see [overlays/README.md](../overlays/README.md)).
   `HOST_RECIPE_DIR` in `host.env` points at that checkout.
2. **Weights, one local copy per host.** Copy the hybrid checkpoint to a local drive on each host and verify
   sha256 of all 288 files against the source. On the CMP host use an NVMe drive and set the read cap in
   `host.env`. No NFS.
3. **Images.** On both hosts: `docker build -t glm53-multinode:base -f image/Dockerfile.base image/`. For the
   Blackwell host, rebuild the two extension libraries for sm_120 and build the second image (next section).
4. **`host.env` on each host** from `host.env.example`: the leader's address on the link (the same on both
   hosts), this host's own address and interface, image, paths, memory limit, utilization, loader threads
   (Blackwell host), read cap (CMP host).
5. **Dry run each node**: `DRYRUN=1 ./node.sh layouts/pp5bw30.env host.env <rank>` prints the layers the node
   takes and the full `docker run`, and fails on a missing patch file. A layout picks its patch set with `MOUNTS`
   (default `mounts.list`) and can add switches to the node environment with `EXTRA_ENV`; `layouts/pp5bw30-jor.env`
   does both.
6. **Start, node 0 first.** On the Blackwell host: `./node.sh layouts/pp5bw30.env host.env 0`. Within a few
   seconds, on the CMP host: ranks 1, 2, 3 and 4. All nodes load in parallel; the leader's API answers
   `/health` when every stage is up (6 to 11 minutes from local copies).
7. **Check before serving.** `curl http://<API_HOST>:<API_PORT>/health` on the leader, a one-token completion, then
   the quality gate against TP8 goldens and the structured-output probe
   ([docs/quality-floor.md](../docs/quality-floor.md)). Expect about 90.2% agreement.
8. **Stop**: `docker rm -f glm53-<TAG>-n<rank>` for every node, CMP nodes and leader alike.

## The sm_120 image

The pinned image's compiled extension libraries were built for sm_80 / 86 / 89, and PTX JIT does not save a
Blackwell stage: `rms_norm`, `silu_and_mul` and the Marlin W8A16 path fail with "no kernel image is available",
and `moe_align_block_size` silently does nothing. Of the operators probed only the NVFP4 Marlin kernel ran from PTX.
So the Blackwell node
needs `_C_stable_libtorch` and `_moe_C_stable_libtorch` rebuilt for `TORCH_CUDA_ARCH_LIST="8.0 12.0"` from the
source the image was built from ([wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) at
`29e66dad461b11caae6039f510eaa7913f9dfb19`), inside the image itself (its nvcc 13.0 and torch 2.13). CPU only,
about 24 minutes on 30 cores:

```bash
git clone https://github.com/wtdcode/vllm-backport && git -C vllm-backport checkout 29e66dad461b11caae6039f510eaa7913f9dfb19
SRC=$PWD/vllm-backport OUT=$PWD/build image/build-sm120.sh
cp build/out/_C_stable_libtorch.abi3.so build/out/_moe_C_stable_libtorch.abi3.so image/
docker build -t glm53-multinode:sm120 -f image/Dockerfile.sm120 image/
```

The Python tree is untouched, so both images take the same patch mounts. The CMP nodes keep the original libraries.
After the rebuild, check the five operators above against torch references on a Blackwell before loading the model;
the model-level proof is the quality gate.

## Things that bite

- **Autotune deadlock.** With sm_120 and sm_80 ranks in one engine the sm_120 ranks enter the flashinfer autotune
  broadcast while the sm_80 ranks wait in the pipeline warm-up, and the load hangs. `--no-enable-flashinfer-autotune`
  on every node (it is in the layout files).
- **Per-node views are required.** The fork refuses an index that names shards it cannot find, and loads every
  shard it does find. `view-build.py` gives each node only its layers.
- **One layout file for all nodes.** `PART`, `TP`, `PP` and the speculative config must be identical everywhere.
- **`--gpus` quoting.** A device list with a comma needs the inner quotes: `--gpus '"device=0,1"'`. `node.sh` does it.
- **The first long prompt after a load is slow once** (27 s against 20 s for the same 83K prompt two minutes later).
- **The PipeSpec counter line is logged by the leader** (node 0), not by the last stage.
- **Power.** Each Blackwell draws about 360 to 390 W in a cold prefill and 250 to 275 W in single-stream decode, on top of the
  CMP host's draw.
- **The API has no authentication.** `API_HOST` defaults to loopback; put anything wider behind your own gate.
- **The checkpoint is abliterated**; add the safeguards your use needs.

## What was and was not measured

Measured: both layout files, with the environment of this directory, on the hardware above (2026-11-03 to
2026-11-05), with the patch set of `mounts-g4.list` or a part of it, as each number in
[docs/outcomes.md](../docs/outcomes.md) says. On 2026-11-05 the pp5bw30 layout also ran with the set of `mounts.list`
(g5) and with that of `mounts-jor.list` plus the jor switches (`pp5bw30-jor.env`); l62b has not run with g5 or jor.
Not measured: other cards as the fast stage, a switched or slower link, TCP instead of RDMA between the hosts,
more than two hosts. The launcher here is the measured one with site paths and operational plumbing removed; its
dry run was compared against the original's mounts, layer ranges and arguments, and it has not itself started an
engine.
