import json, os, glob
import pandas as pd

rows = []
for f in glob.glob("_compare_out/*.json"):
    with open(f) as fp:
        r = json.load(fp)
    img = os.path.basename(f).split("_k")[0].replace("cmp_", "")
    m = r["metrics"]
    rows.append({
        "image": img,
        "deletion_auc_shap": m["deletion_auc_SHAP"],
        "deletion_auc_bo": m["deletion_auc_BO"],
        "insertion_auc_shap": m["insertion_auc_SHAP"],
        "insertion_auc_bo": m["insertion_auc_BO"],
        "spearman": m["spearman"],
        "L1": m["L1"],
        "coverage": m["coverage_diff_le_tail"],
    })

df = pd.DataFrame(rows)
df.to_csv("summary_results.csv", index=False)
print(df)
