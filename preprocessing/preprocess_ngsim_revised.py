#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PREDICTOR_FEATURES = ["mean_speed","std_speed","std_acc","mean_space_headway"]

def safe_mode(s):
    s = s.dropna()
    if len(s) == 0:
        return "unknown"
    m = s.mode()
    return str(m.iloc[0]) if len(m) else str(s.iloc[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/NGSIM/ngsim_raw.csv")
    ap.add_argument("--out-dir", default="data/NGSIM/revised")
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-frames", type=int, default=10)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.input, low_memory=False)

    req = ["vehicle_id","v_vel","v_acc","lane_id","space_headway"]
    missing = [c for c in req if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw NGSIM columns: {missing}")

    rows = []
    for vid, g in raw.groupby("vehicle_id", sort=True):
        if len(g) < a.min_frames:
            continue
        v = pd.to_numeric(g["v_vel"], errors="coerce")
        acc = pd.to_numeric(g["v_acc"], errors="coerce")
        sh = pd.to_numeric(g["space_headway"], errors="coerce")
        lane = pd.to_numeric(g["lane_id"], errors="coerce").dropna()
        lane = lane[lane > 0]
        rows.append({
            "vehicle_id":vid,
            "mean_speed":float(v.mean()),
            "std_speed":float(v.std(ddof=0)),
            "std_acc":float(acc.std(ddof=0)),
            "mean_space_headway":float(sh.mean()),
            "num_frames":int(len(g)),
            "num_unique_lanes":int(lane.nunique()),
            "dominant_lane":float(lane.mode().iloc[0]) if len(lane) and len(lane.mode()) else np.nan,
            "v_class":safe_mode(g["v_class"]) if "v_class" in g else "unknown",
            "location":safe_mode(g["location"]) if "location" in g else "unknown",
            "section_id":safe_mode(g["section_id"]) if "section_id" in g else "unknown",
        })

    feat = pd.DataFrame(rows).replace([np.inf,-np.inf], np.nan)
    feat["target"] = (feat["num_unique_lanes"] > 1).astype(int)
    feat = feat.dropna(subset=PREDICTOR_FEATURES).reset_index(drop=True)

    tr_idx, te_idx = train_test_split(
        np.arange(len(feat)),
        test_size=a.test_fraction,
        random_state=a.seed,
        stratify=feat["target"].to_numpy(),
    )
    train = feat.iloc[tr_idx].copy().reset_index(drop=True)
    test = feat.iloc[te_idx].copy().reset_index(drop=True)

    xsc = StandardScaler()
    Xtr = xsc.fit_transform(train[PREDICTOR_FEATURES].astype(float))
    Xte = xsc.transform(test[PREDICTOR_FEATURES].astype(float))

    all_groups = feat["location"].astype(str)+"::"+feat["section_id"].astype(str)
    group_map = {g:i for i,g in enumerate(sorted(all_groups.unique()))}
    train["client_id"] = (train["location"].astype(str)+"::"+train["section_id"].astype(str)).map(group_map).astype(int)
    test["client_id"] = (test["location"].astype(str)+"::"+test["section_id"].astype(str)).map(group_map).astype(int)

    tr = pd.DataFrame(Xtr, columns=["x1","x2","x3","x4"])
    te = pd.DataFrame(Xte, columns=["x1","x2","x3","x4"])
    tr["target"] = train["target"].to_numpy()
    te["target"] = test["target"].to_numpy()
    tr["client_id"] = train["client_id"].to_numpy()
    te["client_id"] = test["client_id"].to_numpy()

    mcols = ["vehicle_id","location","section_id","num_frames","num_unique_lanes","dominant_lane","v_class","target","client_id"]
    trm = train[mcols]
    tem = test[mcols]

    feat.to_csv(out/"ngsim_revised_vehicle_features.csv", index=False)
    tr.to_csv(out/"ngsim_revised_train.csv", index=False)
    te.to_csv(out/"ngsim_revised_test.csv", index=False)
    trm.to_csv(out/"ngsim_revised_train_metadata.csv", index=False)
    tem.to_csv(out/"ngsim_revised_test_metadata.csv", index=False)

    summary = {
        "sample_unit":"vehicle trajectory",
        "raw_rows":int(len(raw)),
        "vehicle_samples":int(len(feat)),
        "train_rows":int(len(tr)),
        "test_rows":int(len(te)),
        "predictor_features":PREDICTOR_FEATURES,
        "label_definition":"target=1 if vehicle occupies more than one valid lane_id",
        "vehicle_overlap":int(len(set(trm.vehicle_id)&set(tem.vehicle_id))),
        "train_target_counts":{str(k):int(v) for k,v in tr.target.value_counts().sort_index().items()},
        "test_target_counts":{str(k):int(v) for k,v in te.target.value_counts().sort_index().items()},
        "natural_client":"dominant location::section_id",
        "train_clients":int(tr.client_id.nunique()),
        "test_clients":int(te.client_id.nunique()),
        "seed":a.seed
    }
    (out/"ngsim_revised_preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
