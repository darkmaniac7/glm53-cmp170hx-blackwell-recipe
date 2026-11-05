#!/bin/bash
# node.sh: start ONE node of a multi-node GLM-5.3 engine (vLLM multiprocess executor across hosts, no ray).
#
#   node.sh <layout.env> <host.env> <node-rank>          start the node
#   DRYRUN=1 node.sh <layout.env> <host.env> <node-rank> print the docker command, check every mount source, start nothing
#
# One node = one tensor-parallel group = one container. Run it once per node, on the host that owns the node's GPUs,
# node 0 first: node 0 is the leader, it holds the first pipeline stage and serves the API. Start the other nodes within
# a few seconds. Every node must use the same layout file.
#
# What goes into the container (the variables are documented in layouts/*.env and host.env.example):
#   image       HOST_IMAGE          the sm_80 engine image on a CMP host, the sm_120 build on a Blackwell host (image/)
#   GPUs        NODE_<rank>         indices or UUIDs of the node's TP group
#   node rank   <node-rank>         0 .. NNODES-1, in pipeline order
#   master      MASTER_ADDR         node 0's address on the fast link; MASTER_PORT
#   partition   PART                layers per pipeline stage (VLLM_PP_LAYER_PARTITION), the same on every node
#   weights     a per-node "view" (view-build.py): symlinks to only the shards of the node's layers
#   mounts      mounts.list         the patch files, read-only over the image's files; the same list on every node
#   env         node.env            engine and NCCL environment, plus the per-host interface and address
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 3 ] || { echo "usage: node.sh <layout.env> <host.env> <node-rank>"; exit 2; }
# shellcheck disable=SC1090
source "$1"; source "$2"; NR=$3
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

nodevar="NODE_$NR"; GPUS=${!nodevar:?layout has no NODE_$NR}
[ "$NR" -lt "$NNODES" ] || { echo "node rank $NR >= NNODES $NNODES"; exit 2; }

# The node's stages and layers. With PP stages on NNODES nodes every node holds PP / NNODES consecutive stages.
IFS=, read -r -a P <<< "$PART"
[ ${#P[@]} -eq "$PP" ] || { echo "PART must list $PP stages"; exit 2; }
SPN=$((PP / NNODES)); S0=$((NR * SPN)); S1=$((S0 + SPN - 1))
A=0; for ((i = 0; i < S0; i++)); do A=$((A + P[i])); done
B=$A; for ((i = S0; i <= S1; i++)); do B=$((B + P[i])); done; B=$((B - 1))
FLAGS=(); [ "$S0" -eq 0 ] && FLAGS+=(embed)
if [ "$S1" -eq $((PP - 1)) ]; then FLAGS+=(head); [ -n "${SPEC_JSON:-}" ] && FLAGS+=(mtp); fi

# Per-node view of the weights: the fork loads every *.safetensors it finds and refuses an index that names absent
# shards, so each node gets a directory with links to its own shards and a filtered index.
VIEW=$HOST_VIEW_DIR/$TAG-n$NR
if [ "${DRYRUN:-0}" = 1 ]; then echo "would build view $VIEW: layers $A-$B ${FLAGS[*]:-}"
else python3 "$HERE/view-build.py" "$VIEW" "$HOST_MODEL_DIR" "$A" "$B" "${FLAGS[@]}"; fi

# Patch mounts: <file in the recipe's patches/vllm> <target below the vllm package>
M=(); bad=0
while read -r f t; do
  case $f in ''|'#'*) continue;; esac
  [ -f "$HOST_RECIPE_DIR/patches/vllm/$f" ] || { echo "missing patch file $HOST_RECIPE_DIR/patches/vllm/$f"; bad=1; }
  M+=(-v "$HOST_RECIPE_DIR/patches/vllm/$f:$VLLM/$t:ro")
done < "$HERE/mounts.list"
[ "$bad" -eq 0 ] || exit 3

SPEC=(); [ -n "${SPEC_JSON:-}" ] && SPEC=(--speculative-config "$SPEC_JSON")
HEADLESS=(); [ "$NR" -gt 0 ] && HEADLESS=(--headless)
# The consumer NVMe that holds the weights gets a read cap (see README: 48 to 81 C in two and a half minutes without it).
IO=(); [ -n "${HOST_READ_BPS_DEVICE:-}" ] && IO=(--device-read-bps "$HOST_READ_BPS_DEVICE:${HOST_READ_BPS:-450mb}")
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-} ${HOST_EXTRA_ARGS:-})
[ -n "${HOST_LOADER_JSON:-}" ] && EXTRA+=(--model-loader-extra-config "$HOST_LOADER_JSON")

C=glm53-$TAG-n$NR
CMD=(docker run -d --name "$C" --hostname "$C" --init --gpus "\"device=$GPUS\"" --network host --ipc host
  --shm-size 32g --pids-limit 8192 --memory "$HOST_MEM" --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
  "${IO[@]}"
  --env-file "$HERE/node.env" -e "VLLM_HOST_IP=$HOST_ADDR" -e "NCCL_SOCKET_IFNAME=$HOST_IFNAME" -e "GLOO_SOCKET_IFNAME=$HOST_IFNAME"
  -e "VLLM_PP_LAYER_PARTITION=$PART" -e "CMP_PIPESPEC=${CMP_PIPESPEC:-1}" -e NCCL_DEBUG=WARN
  -v "$VIEW:/model:ro" -v "$HOST_MODEL_DIR:$HOST_MODEL_DIR:ro" -v "$HOST_CACHE_DIR:/cache" -v "$HOST_LOG_DIR:/logs" "${M[@]}"
  "$HOST_IMAGE" /model --served-model-name "${SERVED_MODEL_NAME:-glm-5.3}"
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --distributed-executor-backend mp --nnodes "$NNODES" --node-rank "$NR" --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29531}" "${HEADLESS[@]}"
  --attention-backend TRITON_MLA_SPARSE --max-model-len "$MAX_LEN" --gpu-memory-utilization "$HOST_UTIL" --kv-cache-dtype fp8_ds_mla
  --trust-remote-code --enable-chunked-prefill --max-num-seqs "${MAX_SEQS:-8}" --max-num-batched-tokens "${MAX_BATCHED:-2048}"
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice "${SPEC[@]}" "${EXTRA[@]}")

if [ "${DRYRUN:-0}" = 1 ]; then
  echo "node $NR of $NNODES: stages $S0-$S1, layers $A-$B ${FLAGS[*]:-}, gpus $GPUS, ${#M[@]} mount args"
  printf ' %q' "${CMD[@]}"; echo
  exit 0
fi
mkdir -p "$HOST_CACHE_DIR" "$HOST_LOG_DIR"
docker rm -f "$C" > /dev/null 2>&1 || true
"${CMD[@]}" > /dev/null
echo "started $C: node $NR of $NNODES, stages $S0-$S1, layers $A-$B, gpus $GPUS, master $MASTER_ADDR"
