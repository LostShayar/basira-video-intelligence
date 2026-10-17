#!/usr/bin/env bash
# One-time GitHub setup: labels, milestones and Week 1 issues.
# Needs the GitHub CLI: https://cli.github.com  (run `gh auth login` first)
# Run from inside the repo folder after the first push.
set -euo pipefail

# Labels
gh label create feature  --color 1D76DB --description "New capability" --force
gh label create test     --color 0E8A16 --description "Tests"          --force
gh label create docs     --color C5DEF5 --description "Documentation"  --force
gh label create infra    --color 5319E7 --description "CI, Docker, setup" --force

# Milestones (one per release)
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
for m in "v0.1 Perception" "v0.2 Events and product" "v0.3 Intelligence" "v1.0 Operator copilot"; do
  gh api "repos/$repo/milestones" -f title="$m" >/dev/null 2>&1 || true
done

# Week 1 issues
ms="v0.1 Perception"
gh issue create --milestone "$ms" --label infra   --title "Day 1: repo setup, environment, sample videos" \
  --body "Done when: repo public, CI green, 3 sample clips in data/ (not committed)."
gh issue create --milestone "$ms" --label feature --title "Day 2: multi-stream ingestion with bounded queues" \
  --body "Done when: 3 videos play side by side; queue drops stale frames; FPS printed per camera."
gh issue create --milestone "$ms" --label feature --title "Day 3: YOLO detection + ByteTrack tracking" \
  --body "Done when: each person/vehicle keeps one ID across frames on every camera."
gh issue create --milestone "$ms" --label feature --title "Day 4: zones and line crossing from config" \
  --body "Done when: in/out counts and zone occupancy drawn on screen, polygons read from cameras.yaml."
gh issue create --milestone "$ms" --label feature --title "Day 5: loitering, crowd density, face blur" \
  --body "Done when: loiter alert fires after N seconds; faces blurred in output frames."
gh issue create --milestone "$ms" --label test    --title "Day 6: tiled view, GIF, unit tests, tag v0.1" \
  --body "Done when: tests for zone/dwell logic pass in CI; demo GIF in README; v0.1 release published."
echo "Done. Open the Issues tab to see Week 1."
