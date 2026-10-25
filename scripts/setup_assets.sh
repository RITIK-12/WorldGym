#!/usr/bin/env bash
# Sparse-clone MuJoCo Menagerie into assets/menagerie (only the robots we use).
# Add a robot later with: git -C assets/menagerie sparse-checkout add <dir>
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets
if [ ! -d assets/menagerie/.git ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/menagerie
fi
git -C assets/menagerie sparse-checkout set franka_emika_panda robotiq_2f85 "$@"
