#!/usr/bin/env bash
# Re-runs the topic-control revision (steps 1-4). Read-only for results/topic_control/.
# Steps 5-7 need completed human annotations and are run separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY src/topic_control_revision/01_verify_and_diagnose.py
$PY src/topic_control_revision/02_sensitivity_and_tables.py
$PY src/topic_control_revision/03_figures.py
# Step 4 redraws the validation sample; it refuses to overwrite filled sheets.
$PY src/topic_control_revision/04_prepare_validation.py
