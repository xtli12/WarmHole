#!/bin/bash
# One-time environment setup on the server (runs in the background; log in $BH/logs/setup.log)
BH=${BH:-/data/blackhole}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$BH/uv_cache UV_PYTHON_INSTALL_DIR=$BH/uv_python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
# route github through a proxy for this process tree only (no change to the user's git config)
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://gh-proxy.com/https://github.com/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/
export GIT_LFS_SKIP_SMUDGE=1
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple
step() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $BH/logs $BH/openpi_data/big_vision $BH/venvs $BH/libero_cfg

step "pi0_libero checkpoint download (GCS, range-parallel) in background"
(for k in 1 2 3; do
   [ -f $BH/logs/pi0_libero.complete ] && break
   python3 $BH/code/dl_fast.py openpi-assets checkpoints/pi0_libero/ $BH/openpi_data/openpi-assets $BH/logs/pi0_libero.tmp 32
 done > $BH/logs/dl_pi0.log 2>&1) &

step "paligemma tokenizer"
curl -sSL --retry 10 -o $BH/openpi_data/big_vision/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model; ls -la $BH/openpi_data/big_vision

step "openpi source (commit 215abfb) + LIBERO submodule"
[ -d $BH/openpi/.git ] || git clone https://github.com/Physical-Intelligence/openpi.git $BH/openpi
cd $BH/openpi && git checkout -q 215abfb && git submodule update --init --recursive && git log -1 --oneline
ls third_party/libero | head -3

step "openpi venv (Python 3.11, exact lock versions via the Tsinghua mirror)"
uv python install 3.11
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv export --frozen --no-hashes --format requirements-txt --no-emit-project --no-emit-workspace > $BH/code/openpi_req.txt
wc -l $BH/code/openpi_req.txt; grep -n "git+" $BH/code/openpi_req.txt
VIRTUAL_ENV=$BH/openpi/.venv uv pip install -r $BH/code/openpi_req.txt --index-url $TUNA --index-strategy unsafe-best-match
VIRTUAL_ENV=$BH/openpi/.venv uv pip install --no-deps -e . -e packages/openpi-client --index-url $TUNA
.venv/bin/python -c "import jax, flax, openpi, transformers; print('OPENPI OK', jax.__version__, flax.__version__, jax.devices())"

step "LIBERO client venv (Python 3.8)"
VENV=$BH/venvs/libero
[ -x $VENV/bin/python ] || uv venv --python 3.8 $VENV
REQ=$BH/code/req_client.txt
grep -v '^ *#' examples/libero/requirements.txt | sed '/^ *$/d' \
  | sed -e 's/^torch==1.11.0+cu113/torch==1.11.0+cpu/' -e 's/^torchaudio==0.11.0+cu113/torchaudio==0.11.0+cpu/' \
        -e 's/^torchvision==0.12.0+cu113/torchvision==0.12.0+cpu/' > $REQ
printf "hydra-core==1.2.0\neasydict==1.9\neinops==0.4.1\nbddl==1.0.1\nfuture==0.18.2\ncloudpickle==2.1.0\ngym==0.25.2\nwebsockets==11.0.3\nmsgpack==1.0.5\n" >> $REQ
VIRTUAL_ENV=$VENV uv pip install -r $REQ --index-url $TUNA --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
cat > $BH/libero_cfg/config.yaml <<EOC
benchmark_root: $BH/openpi/third_party/libero/libero/libero
bddl_files: $BH/openpi/third_party/libero/libero/libero/bddl_files
init_states: $BH/openpi/third_party/libero/libero/libero/init_files
datasets: $BH/openpi/third_party/libero/libero/datasets
assets: $BH/openpi/third_party/libero/libero/libero/assets
EOC
PYTHONPATH=$BH/openpi/third_party/libero:$BH/openpi/packages/openpi-client/src LIBERO_CONFIG_PATH=$BH/libero_cfg \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=3 $VENV/bin/python -c "
from libero.libero.envs import OffScreenRenderEnv; from libero.libero import benchmark, get_libero_path; import os, time
s = benchmark.get_benchmark_dict()['libero_spatial'](); t = s.get_task(7)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path('bddl_files'), t.problem_folder, t.bddl_file), camera_heights=256, camera_widths=256)
env.seed(0); env.reset(); env.set_init_state(s.get_task_init_states(7)[0]); t0 = time.time()
[env.step([0]*6+[-1]) for _ in range(50)]; print('LIBERO OK step ms', round((time.time()-t0)*20, 1))" && touch $BH/logs/client.ok

step "overlay (float attention mask + attention capture)"
rm -rf $BH/code/overlay && mkdir -p $BH/code/overlay && cp -r $BH/openpi/src/openpi $BH/code/overlay/
$BH/openpi/.venv/bin/python $BH/code/patch_overlay.py $BH/code/overlay

step "pi05_libero checkpoint (hf-mirror copy of the official files, verified against the GCS listing)"
D=$BH/openpi_data/openpi-assets/checkpoints/pi05_libero
for k in 1 2 3 4; do
  [ -f $BH/logs/pi05_libero.complete ] && break
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=$BH/hf $BH/openpi/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('bf-jeon/pi05_libero', local_dir='$D', max_workers=8)
print('hf download finished')"
  $BH/openpi/.venv/bin/python $BH/code/verify_ckpt.py openpi-assets checkpoints/pi05_libero/ $D $BH/logs/pi05_libero.complete
done
step "wait for pi0 and verify"
wait
$BH/openpi/.venv/bin/python $BH/code/verify_ckpt.py openpi-assets checkpoints/pi0_libero/ $BH/openpi_data/openpi-assets/checkpoints/pi0_libero $BH/logs/pi0_libero.complete
ls $BH/logs/*.complete
step "SETUP DONE"
