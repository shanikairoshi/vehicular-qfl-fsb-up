#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PREDICTOR_FEATURES = ["mean_v","std_v","displacement","num_frames"]
LABEL_FEATURES = ["std_acc","mean_abs_acc_lat","mean_abs_acc_tan"]

def get_tables(con):
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

def extract_table(con, table, site, min_frames):
    query = (
        'SELECT OBJID, TIMESTAMP, UTM_X, UTM_Y, V, ACC, ACC_LAT, ACC_TAN, CLASS '
        f'FROM "{table}" ORDER BY OBJID, TIMESTAMP'
    )
    df = pd.read_sql_query(query, con)
    rows = []
    for objid, g in df.groupby("OBJID", sort=False):
        if len(g) < min_frames:
            continue
        v = pd.to_numeric(g["V"], errors="coerce")
        acc = pd.to_numeric(g["ACC"], errors="coerce")
        alat = pd.to_numeric(g["ACC_LAT"], errors="coerce")
        atan = pd.to_numeric(g["ACC_TAN"], errors="coerce")
        x = pd.to_numeric(g["UTM_X"], errors="coerce")
        y = pd.to_numeric(g["UTM_Y"], errors="coerce")
        good = x.notna() & y.notna()
        if good.sum() < 2:
            continue
        xv, yv = x[good], y[good]
        displacement = float(np.hypot(xv.iloc[-1]-xv.iloc[0], yv.iloc[-1]-yv.iloc[0]))
        cm = g["CLASS"].dropna().mode()
        cls = str(cm.iloc[0]) if len(cm) else "unknown"
        rows.append({
            "site":site,
            "recording":table,
            "objid":int(objid),
            "trajectory_uid":f"{site}::{table}::{int(objid)}",
            "class":cls,
            "mean_v":float(v.mean()),
            "std_v":float(v.std(ddof=0)),
            "std_acc":float(acc.std(ddof=0)),
            "mean_abs_acc_lat":float(alat.abs().mean()),
            "mean_abs_acc_tan":float(atan.abs().mean()),
            "displacement":displacement,
            "num_frames":int(len(g)),
        })
    return pd.DataFrame(rows)

def split_recordings(feat, test_fraction, seed):
    rng = np.random.default_rng(seed)
    test_keys = set()
    recs = feat[["site","recording"]].drop_duplicates()
    for site, g in recs.groupby("site"):
        arr = np.array(sorted(g["recording"].unique()))
        rng.shuffle(arr)
        n_test = max(1, int(round(len(arr)*test_fraction)))
        for r in arr[:n_test]:
            test_keys.add((site,r))
    mask = np.array([(s,r) in test_keys for s,r in zip(feat.site, feat.recording)])
    return feat.loc[~mask].copy(), feat.loc[mask].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", default=None)
    ap.add_argument("--out-dir", default="data/OpenDD/revised")
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-frames", type=int, default=10)
    a = ap.parse_args()

    dbs = a.db or [
        "data/opendd_v3-rdb3/rdb3/trajectories_rdb3_v3.sqlite",
        "data/opendd_v3-rdb5/rdb5/trajectories_rdb5_v3.sqlite",
    ]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parts = []
    raw_rows = 0
    for dbp in dbs:
        p = Path(dbp)
        site = p.parent.name
        con = sqlite3.connect(str(p))
        tables = get_tables(con)
        print(f"[{site}] recordings={len(tables)}")
        for i,t in enumerate(tables,1):
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            raw_rows += int(n)
            f = extract_table(con, t, site, a.min_frames)
            if len(f):
                parts.append(f)
            print(f"[{site}] {i}/{len(tables)} {t}: frames={n}, objects={len(f)}")
        con.close()

    feat = pd.concat(parts, ignore_index=True).replace([np.inf,-np.inf], np.nan)
    feat = feat.dropna(subset=PREDICTOR_FEATURES + LABEL_FEATURES).reset_index(drop=True)
    if feat["trajectory_uid"].duplicated().any():
        raise ValueError("Duplicate trajectory_uid detected.")

    train, test = split_recordings(feat, a.test_fraction, a.seed)

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

    all_keys = feat["site"].astype(str)+"::"+feat["recording"].astype(str)
    rec_map = {r:i for i,r in enumerate(sorted(all_keys.unique()))}
    train["client_id"] = (train["site"].astype(str)+"::"+train["recording"].astype(str)).map(rec_map).astype(int)
    test["client_id"] = (test["site"].astype(str)+"::"+test["recording"].astype(str)).map(rec_map).astype(int)

    tr = pd.DataFrame(Xtr, columns=["x1","x2","x3","x4"])
    te = pd.DataFrame(Xte, columns=["x1","x2","x3","x4"])
    tr["target"] = train["target"].to_numpy()
    te["target"] = test["target"].to_numpy()
    tr["client_id"] = train["client_id"].to_numpy()
    te["client_id"] = test["client_id"].to_numpy()

    mcols = ["site","recording","objid","trajectory_uid","class","num_frames","derived_score","target","client_id"]
    trm = train[mcols].reset_index(drop=True)
    tem = test[mcols].reset_index(drop=True)

    feat.to_csv(out/"opendd_revised_object_features.csv", index=False)
    tr.to_csv(out/"opendd_revised_train.csv", index=False)
    te.to_csv(out/"opendd_revised_test.csv", index=False)
    trm.to_csv(out/"opendd_revised_train_metadata.csv", index=False)
    tem.to_csv(out/"opendd_revised_test_metadata.csv", index=False)

    train_rec = set(zip(trm.site,trm.recording))
    test_rec = set(zip(tem.site,tem.recording))
    summary = {
        "sample_unit":"object trajectory",
        "raw_frame_rows":int(raw_rows),
        "object_trajectories":int(len(feat)),
        "train_rows":int(len(tr)),
        "test_rows":int(len(te)),
        "predictor_features":PREDICTOR_FEATURES,
        "label_features":LABEL_FEATURES,
        "label_definition":"train-standardized dynamics score > train median",
        "recording_overlap":int(len(train_rec & test_rec)),
        "trajectory_overlap":int(len(set(trm.trajectory_uid)&set(tem.trajectory_uid))),
        "train_target_counts":{str(k):int(v) for k,v in tr.target.value_counts().sort_index().items()},
        "test_target_counts":{str(k):int(v) for k,v in te.target.value_counts().sort_index().items()},
        "natural_client":"site::recording",
        "train_clients":int(tr.client_id.nunique()),
        "test_clients":int(te.client_id.nunique()),
        "class_counts":{str(k):int(v) for k,v in feat["class"].value_counts().items()},
        "seed":a.seed
    }
    (out/"opendd_revised_preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
