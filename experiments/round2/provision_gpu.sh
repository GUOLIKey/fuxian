#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p gpu-direct
export gpu_dir="$PWD/gpu-direct"
seq 0 135 | xargs -P 16 -I '{}' /bin/bash -c '
  set -e
  piece=$1; total=2267330315; length=16777216
  offset=$((piece * length)); end=$((offset + length - 1))
  if [ "$end" -ge "$total" ]; then end=$((total - 1)); fi
  name=$(printf "%s/cu118.%03d" "$gpu_dir" "$piece")
  for retry in 1 2 3 4; do
      if curl -fsSL --connect-timeout 12 --max-time 110 -r "$offset-$end" -o "$name.partial" "https://download-r2.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl?fuxian-server-r2=$piece" && [ "$(wc -c < "$name.partial")" -eq "$((end-offset+1))" ]; then
          mv "$name.partial" "$name"
          printf "chunk %d complete\n" "$piece"
          exit 0
      fi
  done
  exit 1' _ '{}'
cd "$gpu_dir"
wheel=torch-2.0.1+cu118-cp311-cp311-linux_x86_64.whl
cat cu118.[0-9][0-9][0-9] > "$wheel"
echo "143b6c658c17d43376e2dfbaa2c106d35639d615e5e8dec4429cf1e510dd8d61  $wheel" | sha256sum -c -
cd ..
mkdir gpu-install-lock
py=/root/autodl-tmp/fuxian-run/reproduction_runs/env-fedvade/bin/python
"$py" -m venv --system-site-packages env-gpu
env-gpu/bin/python -m pip install --ignore-installed --no-deps --no-index "$gpu_dir/$wheel"
env-gpu/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name()); assert torch.cuda.is_available(); print(torch.ones(1,device="cuda"))'
env-gpu/bin/python -m pip freeze > gpu-freeze.txt
touch gpu-ready
