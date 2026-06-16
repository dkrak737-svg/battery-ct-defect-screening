#!/bin/bash
# RunPod-side setup: untar uploaded data, install deps, ready to tile+train.
# Run ONCE after uploading tars/ and code/ to /workspace/battery-ct-security/.
set -euo pipefail

PROJ=/workspace/battery-ct-security
DATA=$PROJ/data

cd "$PROJ"

# ---- Python env ----
if [ ! -d "$PROJ/.venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ---- Untar uploaded battery archives -> data/cropped/ ----
mkdir -p "$DATA/cropped"
cp "$DATA/tars/labels.jsonl" "$DATA/" 2>/dev/null || true

N=0
for t in "$DATA"/tars/*.tar; do
    [ -f "$t" ] || continue
    tar -xf "$t" -C "$DATA/cropped/"
    N=$((N+1))
    [ $((N % 10)) -eq 0 ] && echo "  untarred $N..."
done
echo "Untar done: $(ls "$DATA"/cropped/ | wc -l) battery folders"

# ---- Tile (one-shot) -> data/yolo_data/ ----
python preprocessing/tile_dataset.py --clean

echo ""
echo "Setup complete. To train (with auto-Stop):"
echo "  nohup bash vision/train_and_stop.sh > train.out &"
