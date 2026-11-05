#!/bin/bash
# build-sm120.sh: rebuild vLLM's _C_stable_libtorch and _moe_C_stable_libtorch for sm_80 + sm_120 inside the pinned
# engine image (its own nvcc 13.0 and torch 2.13). CPU only, no GPU needed; about 24 minutes on 30 cores.
#
#   SRC=/path/to/vllm-backport OUT=/path/to/build ./build-sm120.sh
#
# SRC is a checkout of https://github.com/wtdcode/vllm-backport at 29e66dad461b11caae6039f510eaa7913f9dfb19, the
# commit the pinned image was built from (compare its vllm/ python tree with the image's before trusting the build).
# The two libraries land in $OUT/out; copy them next to Dockerfile.sm120.
set -eu
: "${SRC:?set SRC to the vllm-backport checkout}" "${OUT:?set OUT to a build directory}"
IMG=lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e
JOBS=${JOBS:-30}
mkdir -p "$OUT/out"
docker run --rm --name glm53-sm120-build --network host --cpus "$((JOBS + 2))" --memory "${MEM:-96g}" \
  -e JOBS="$JOBS" -v "$SRC:/src" -v "$OUT:/build" --entrypoint bash "$IMG" -c '
set -x
cd /build
which git || (apt-get update -qq && apt-get install -y -qq git)
python3 -m pip install -q "cmake>=3.26,<4" 2>&1 | tail -2
export PATH=/usr/local/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0 12.0" MAX_JOBS=$JOBS NVCC_THREADS=2
nice -n 19 cmake -S /src -B /build/b -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH=$(python3 -c "import sys;print(\":\".join(sys.path))") \
  -DFETCHCONTENT_BASE_DIR=/build/deps -DNVCC_THREADS=2 -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
  -DCMAKE_JOB_POOLS:STRING=compile=$JOBS -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /build/configure.log 2>&1
echo CONFIGURE-RC=$?
nice -n 19 cmake --build /build/b --target _C_stable_libtorch _moe_C_stable_libtorch -j $JOBS > /build/build.log 2>&1
rc=$?; echo BUILD-RC=$rc
find /build/b -name "*_stable_libtorch*.so" -newer /build/configure.log -exec cp -v {} /build/out/ \;
exit $rc'
ls -la "$OUT/out"
