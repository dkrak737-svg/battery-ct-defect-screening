#!/bin/bash
# Tar cropped/ into per-battery archives for faster upload to RunPod.
# Run from project root (d:/CT DATA on local; not needed on RunPod).
#
# Result: tars/<battery_key>.tar  (134 files, ~25GB total)
# Plus:   tars/labels.jsonl (copied as-is — small)
set -euo pipefail

CROPPED=cropped
OUT=tars

if [ ! -d "$CROPPED" ]; then
    echo "ERROR: $CROPPED not found. Run preprocess_local.py first."
    exit 1
fi

mkdir -p "$OUT"
cp labels.jsonl "$OUT/" 2>/dev/null || echo "WARN: labels.jsonl not found"

N=0
for d in "$CROPPED"/*/; do
    bkey=$(basename "$d")
    out="$OUT/$bkey.tar"
    if [ -f "$out" ]; then
        echo "skip (exists): $bkey"
        continue
    fi
    # Tar with parent dir so untar restores cropped/<bkey>/ structure
    tar -cf "$out" -C "$CROPPED" "$bkey"
    N=$((N+1))
    [ $((N % 10)) -eq 0 ] && echo "  packed $N batteries..."
done
echo "Done. $(ls "$OUT"/*.tar 2>/dev/null | wc -l) tar files in $OUT/"
du -sh "$OUT"
