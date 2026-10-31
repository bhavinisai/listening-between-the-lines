#!/bin/bash
#SBATCH --job-name=sent_llm_ft
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=logs/sent_llm_ft_%j.out
#SBATCH --error=logs/sent_llm_ft_%j.err

# Fine-tune valence and arousal classifiers on the LLM consensus labels (2 base models each),
# pick the best config per task by CV macro-F1 vs the LLM consensus, then score the full corpus.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
export PYTHONPATH=src/sentiment

R=results/sentiment/llm_ft
mkdir -p $R models
M=$R/metrics.jsonl
rm -f $M

run() { # task tag base lr epochs
  python src/sentiment/finetune_llm_labels.py --task $1 --tag $2 --base $3 --lr $4 --epochs $5 \
      --metrics $M --final_dir models/llm_$1_$2
}
for T in valence arousal; do
  run $T cardiff cardiff_xlmr_multi 2e-5 4
  run $T large xlmr_large 1e-5 5
done

python - <<'PY'
import json, shutil, os
rows = [json.loads(l) for l in open("results/sentiment/llm_ft/metrics.jsonl")]
best = {}
for r in rows:
    if r["task"] not in best or r["cv"]["macro_f1"] > best[r["task"]]["cv"]["macro_f1"]:
        best[r["task"]] = r
for r in rows:
    if best[r["task"]]["tag"] != r["tag"]:
        shutil.rmtree(f"models/llm_{r['task']}_{r['tag']}", ignore_errors=True)
json.dump({t: b["tag"] for t, b in best.items()}, open("results/sentiment/llm_ft/best.json", "w"))
print({t: b["tag"] for t, b in best.items()})
PY
VT=$(python -c "import json;print(json.load(open('$R/best.json'))['valence'])")
AT=$(python -c "import json;print(json.load(open('$R/best.json'))['arousal'])")
python src/sentiment/score_corpus.py --valence_model models/llm_valence_$VT --arousal_model models/llm_arousal_$AT \
    --out results/sentiment/turn_scores_llm.csv
echo "Job finished at $(date)"
