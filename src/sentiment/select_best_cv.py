#!/usr/bin/env python3
"""Print the config tag (file stem) with the highest episode-grouped-CV macro-F1 and write the summary table."""
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{d}/*.csv")):
    if os.path.basename(f) == "summary.csv":
        continue
    x = pd.read_csv(f)
    pred = x[["p_neg", "p_neu", "p_pos"]].values.argmax(1)
    y = x.valence.map({"negative": 0, "neutral": 1, "positive": 2})
    rows.append(dict(config=os.path.basename(f)[:-4], accuracy=accuracy_score(y, pred),
                     macro_f1=f1_score(y, pred, average="macro"), kappa=cohen_kappa_score(y, pred)))
s = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
s.to_csv(f"{d}/summary.csv", index=False)
print(s.round(3).to_string(index=False), file=sys.stderr)
print(s.config.iloc[0])
