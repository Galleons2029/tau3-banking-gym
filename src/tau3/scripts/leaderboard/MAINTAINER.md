# Leaderboard Maintainer Guide

> **Note:** This harness reviews and publishes **text** submissions only. The
> `voice_submissions` array and the `modality: "voice"` entries already in the
> manifest are historical — their schema is still validated, but the tooling to
> review, recompute or backfill voice interaction metrics has been removed.

## Prerequisites

- AWS CLI configured with `tau-bench-ci` profile
- S3 bucket: `sierra-tau-bench-public`
- S3 prefix: `submissions/`

## Handling a Submission PR

When someone opens a PR with a new `submission.json`:

### 1. Review the PR

Check the submission.json manually:
- Directory name follows `{model_name}_{org}_{date}` convention
- `manifest.json` is updated (correct array: `submissions` or `legacy_submissions`;
  `voice_submissions` is historical and takes no new entries)
- `submission_type` is `"standard"` or `"custom"` (if custom: check `methodology.notes` and `references`)
- Contact info is present
- `modality` is `"text"`

### 2. Download their trajectories

The PR description should include a link to externally hosted trajectory files.
Download them locally:

```bash
# Example: download from Google Drive, HuggingFace, etc.
mkdir -p /tmp/review-trajectories
# ... download to /tmp/review-trajectories/
```

The trajectory source should contain one or more `results.json` (or `*.json`)
files, one per domain.

### 3. Run the review script

```bash
python -m tau3.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/

# Or with --upload to also push to S3:
python -m tau3.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/ \
  --upload --aws-profile tau-bench-ci
```

The script will:
1. Parse the PR's `submission.json`
2. Load and validate trajectory files (format, task coverage, trial counts)
3. Recompute metrics and compare against submitted scores
4. Update `submission.json` with `trajectories_available: true` and `trajectory_files`
5. (With `--upload`) Upload trajectories and updated `submission.json` to S3
6. (With `--upload`) Verify the upload (HTTP 200, file sizes match)

### 4. Verify upload separately (optional)

If you uploaded manually or want to re-check:

```bash
python -m tau3.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/ \
  --verify-only --aws-profile tau-bench-ci
```

### 5. Commit and merge

After the script updates `submission.json`, commit the change and merge the PR.
The GitHub Actions workflow will sync `submission.json` and `manifest.json` to S3
(trajectories are excluded from the sync — they were uploaded directly by the script).

## Manual S3 Commands

If you need to do things manually:

```bash
# Upload trajectories
aws s3 sync \
  /local/trajectories/ \
  s3://sierra-tau-bench-public/submissions/<submission-dir>/trajectories/ \
  --profile tau-bench-ci

# Upload submission.json
aws s3 cp \
  web/leaderboard/public/submissions/<submission-dir>/submission.json \
  s3://sierra-tau-bench-public/submissions/<submission-dir>/submission.json \
  --profile tau-bench-ci

# Verify a file exists and check size
aws s3api head-object \
  --bucket sierra-tau-bench-public \
  --key submissions/<submission-dir>/trajectories/<filename> \
  --profile tau-bench-ci \
  --query '[ContentLength, ETag]'

# Delete and re-upload (if corrupted)
aws s3 rm s3://sierra-tau-bench-public/submissions/<dir>/trajectories/<file> --profile tau-bench-ci
aws s3 cp /local/path s3://sierra-tau-bench-public/submissions/<dir>/trajectories/<file> --profile tau-bench-ci
```

## Troubleshooting

### Large trajectory files get corrupted during upload

Files over ~500 MB can get corrupted during S3 multipart upload. Symptoms:
- S3 file size differs from local
- JSON parse errors when fetching from S3

Verify integrity:
```bash
# Compare local vs S3 sizes
local_size=$(stat -f%z /local/path/results.json)
s3_size=$(aws s3api head-object --bucket sierra-tau-bench-public --key <key> --profile tau-bench-ci --query ContentLength --output text)
echo "local: $local_size, s3: $s3_size"
```

If corrupted, delete and re-upload with `aws s3 cp` (not sync).
