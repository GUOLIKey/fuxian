#!/usr/bin/env bash
# Package small result artifacts (no checkpoints, no source copies) from the
# server into a tarball for the GitHub repository.
set -euo pipefail
E=/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro
OUT=/tmp/round7_f01_artifacts.tar.gz
LIST=/tmp/round7_f01_artifacts.list
: > "$LIST"

for f in progress.txt search_plan.json coarse_rankings.json fine_plan.json \
         fine_rankings.json error.txt REPORT.md README.md; do
  [ -f "$E/$f" ] && echo "${f}" >> "$LIST"
done
for d in audit results monitoring scripts; do
  find "$E/$d" -maxdepth 1 -type f >> "$LIST"
done
for stage in priority coarse fine; do
  [ -d "$E/$stage" ] || continue
  for run in "$E/$stage"/*/; do
    [ -d "$run" ] || continue
    for f in config.json result.json rounds.jsonl evaluation_summary.json \
             inference.csv full_checkpoint_manifest.jsonl source.diff \
             resume_verify.json resume_verify_125.json compress.log; do
      [ -f "$run$f" ] && echo "${run#$E/}$f" >> "$LIST"
    done
  done
done
# Driver and evaluation logs are small enough to keep.
for f in "$E"/logs/search_driver.log "$E"/logs/priority_F01_eval.log; do
  [ -f "$f" ] && echo "${f#$E/}" >> "$LIST"
done

tar -C "$E" -czf "$OUT" -T "$LIST"
echo "wrote $OUT"
wc -l "$LIST"
ls -lh "$OUT"
