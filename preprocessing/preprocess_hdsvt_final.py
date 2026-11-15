#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PREDICTOR_FEATURES = ["x_mean_speed_proxy","x_speed_std","x_speed_max","track_length"]
LABEL_FEATURES = ["x_acc_std","x_heading_change","x_curvature_ratio"]

def validate(df):
    req = PREDICTOR_FEATURES + LABEL_FEATURES + ["source_file","trajectory_uid","vehicle_id","vehicle_class"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["trajectory_uid"].duplicated().any():
        raise ValueError("trajectory_uid must be unique.")
    finite = np.isfinite(df[PREDICTOR_FEATURES + LABEL_FEATURES].astype(float)).all(axis=1)
    return df.loc[finite].reset_index(drop=True)

def split_by_source(df, test_fraction, seed):
    groups = np.array(sorted(df["source_file"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups)*test_fraction)))
    test_groups = set(groups[:n_test])
    m = df["source_file"].astype(str).isin(test_groups)
    return df.loc[~m].copy(), df.loc[m].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/HDSVT/processed_nonIID/hdsvt_vehicle_level_features.csv")
    ap.add_argument("--out-dir", default="data/HDSVT/revised")
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = validate(pd.read_csv(a.input))
    train, test = split_by_source(full, a.test_fraction, a.seed)

    ysc = StandardScaler()
    ztr = ysc.fit_transform(train[LABEL_FEATURES].astype(float))
    zte = ysc.transform(test[LABEL_FEATURES].astype(float))
    score_tr, score_te = ztr.sum(axis=1), zte.sum(axis=1)
    threshold = float(np.median(score_tr))
    train["derived_score"] = score_tr
    test["derived_score"] = score_te
    train["target"] = (score_tr > threshold).astype(int)
    test["target"] = (score_te > threshold).astype(int)

    xsc = StandardScaler()
    Xtr = xsc.fit_transform(train[PREDICTOR_FEATURES].astype(float))
    Xte = xsc.transform(test[PREDICTOR_FEATURES].astype(float))

    source_map = {s:i for i,s in enumerate(sorted(full["source_file"].astype(str).unique()))}
    train["client_id"] = train["source_file"].astype(str).map(source_map).astype(int)
    test["client_id"] = test["source_file"].astype(str).map(source_map).astype(int)

    tr = pd.DataFrame(Xtr, columns=["x1","x2","x3","x4"])
    te = pd.DataFrame(Xte, columns=["x1","x2","x3","x4"])
    tr["target"] = train["target"].to_numpy()
    te["target"] = test["target"].to_numpy()
    tr["client_id"] = train["client_id"].to_numpy()
    te["client_id"] = test["client_id"].to_numpy()

    mcols = ["source_file","trajectory_uid","vehicle_id","vehicle_class","derived_score","target","client_id"]
    trm = train[mcols].reset_index(drop=True)
    tem = test[mcols].reset_index(drop=True)

    tr.to_csv(out/"hdsvt_revised_train.csv", index=False)
    te.to_csv(out/"hdsvt_revised_test.csv", index=False)
    trm.to_csv(out/"hdsvt_revised_train_metadata.csv", index=False)
    tem.to_csv(out/"hdsvt_revised_test_metadata.csv", index=False)

    summary = {
        "sample_unit":"trajectory",
        "rows_total":int(len(full)),
        "train_rows":int(len(tr)),
        "test_rows":int(len(te)),
        "predictor_features":PREDICTOR_FEATURES,
        "label_features":LABEL_FEATURES,
        "label_definition":"train-standardized dynamics score > train median",
        "source_overlap":int(len(set(trm.source_file)&set(tem.source_file))),
        "trajectory_overlap":int(len(set(trm.trajectory_uid)&set(tem.trajectory_uid))),
        "train_target_counts":{str(k):int(v) for k,v in tr.target.value_counts().sort_index().items()},
        "test_target_counts":{str(k):int(v) for k,v in te.target.value_counts().sort_index().items()},
        "natural_client":"source_file",
        "train_clients":int(tr.client_id.nunique()),
        "test_clients":int(te.client_id.nunique()),
        "seed":a.seed
    }
    (out/"hdsvt_revised_preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
