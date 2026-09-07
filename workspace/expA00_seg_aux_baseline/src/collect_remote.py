"""Pull every finished remote config from the HF repo and rank them.

Ranking is by OOF AUROC (the CV decision metric we validated) with the held-out
EVC score alongside, because the test set is 12 unseen centres and EVC is the only
external-domain read we have. PPV@90Recall is printed but not used to rank -- with
158 positives its bootstrap IQR (0.11) dwarfs the gaps between models (0.02).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("collect")
REPO = "negichi/rare26-work"
DEST = Path(__file__).resolve().parents[3] / "results" / "remote"


def sync() -> list[str]:
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("results/")]
    names = sorted({f.split("/")[1] for f in files})
    for f in files:
        out = DEST / Path(f).relative_to("results")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(REPO, f, repo_type="dataset"), out)
    return names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = sync()
    rows = []
    for n in names:
        d = DEST / n
        r = {"config": n}
        oof = d / "oof.csv"
        if oof.exists():
            t = pd.read_csv(oof)
            r["oof_auroc"] = roc_auc_score(t.label, t.prob)
            r["oof_auprc"] = average_precision_score(t.label, t.prob)
        m = d / "oof_metrics.json"
        if m.exists():
            r["ppv90"] = json.loads(m.read_text())["overall"]["ppv_90recall"]
        evc = next(iter(d.glob("evc_preds_*.csv")), None)
        if evc is not None:
            e = pd.read_csv(evc)
            r["evc_auroc"] = roc_auc_score(e.label, e.logit)
        r["members"] = len(list(d.glob("fold*.pth")))
        r["all_data"] = (d / "all.pth").exists()
        rows.append(r)
    if not rows:
        logger.info("nothing collected yet")
        return
    df = pd.DataFrame(rows).sort_values("oof_auroc", ascending=False)
    pd.set_option("display.width", 160)
    logger.info("%s", df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    logger.info("\ncollected %d config(s); decision metric = oof_auroc, evc_auroc = external-domain read", len(df))


if __name__ == "__main__":
    main()
