#!/usr/bin/env python3
"""
Unified reviewer-revision experiment driver for vehicular QFL.

Stages
------
5  : Controlled S0-S3 security-profile comparison.
6  : FSB-UP context-field security ablation.
7  : Top-K sparsification sweep.
8  : Top-K + error-feedback sweep.
9  : FP32/INT8/INT4 + proposed methods; optional mdQFL/ResFed literature baselines and TeleSync.
10 : Trace-driven V2X delay/loss/disconnection experiments.
11 : Trace-driven handover/staleness experiments.
12 : Consolidate CSV results, communication accounting and summary figures.

This file is derived from the user's revised S0-S3/FSB-UP notebook and retains
its QNN, SPSA, security and communication routines.  It is intended for
non-interactive execution under Slurm on Deakin HPC.
"""
import argparse
import sys
import time
from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
)

# Notebook globals replaced by CLI paths for HPC execution.
TRAIN_CSV = ""
TEST_CSV = ""
OUT_DIR = ""
N_QUBITS = 4

# Final frozen Stage-9 protocol. Do not tune these constants after inspecting test results.
FINAL_STAGE9_PROTOCOL = "TMC_STAGE9_FROZEN_V1_2026-10-26"
FINAL_SUBSET_SEED = 42


"""
Processed-NGSIM Accuracy-Controlled Predictive Residual QFL
==========================================================

This version is designed for your already processed NGSIM files:

    NGSIM/processed/ngsim_encoded_train.csv
    NGSIM/processed/ngsim_encoded_test.csv

Expected columns:
    x1, x2, x3, x4, target, client_id

The code uses:
    - encoded vehicle-level NGSIM features,
    - target for binary classification,
    - client_id for federated client partitions,
    - explicit SPSA learning rate and perturbation,
    - a compact statevector QNN,
    - tuned QFL baseline,
    - accuracy-controlled predictive residual QFL.

Core idea:
    Instead of every client sending the full local QNN update in every round,
    the server predicts each client's update from previous behaviour. The
    client sends only the residual correction when needed. If prediction error
    is too large, the client falls back to a full update to protect accuracy.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



# =============================================================================
# 1. Configuration
# =============================================================================
#TRAIN_CSV = r"HDSVT\processed\hdsvt_encoded_train.csv"
#TEST_CSV  = r"HDSVT\processed\hdsvt_encoded_test.csv"
@dataclass
class DataConfig:
    train_csv: str = TRAIN_CSV
    test_csv: str = TEST_CSV
    target_col: str = "target"
    client_col: str = "client_id"
    feature_prefix: str = "x"
    n_qubits: int = 4
    values_already_angles: bool = False

    # New: reduce runtime using stratified subsets
    train_subset_size: int = 5000 #10 000
    test_subset_size: int = 1500 #3000
    subset_seed: int = FINAL_SUBSET_SEED
    # For synthetic Dirichlet clients, subset by target only so legacy client_id
    # does not influence which rows enter the experiment.
    stratify_by_client: bool = True

@dataclass
class QNNConfig:
    n_qubits: int = 4
    ansatz_layers: int = 2

    @property
    def n_params(self) -> int:
        # Trainable Ry/Rz per qubit per layer.
        return self.ansatz_layers * self.n_qubits * 2


@dataclass
class SPSAConfig:
    local_iter: int = 12
    learning_rate: float = 0.08
    perturbation: float = 0.12
    lr_decay: float = 0.0
    pert_decay: float = 0.0
    batch_size: int = 48
    max_step_norm: float = 0.35


@dataclass
class PredictiveConfig:
    warmup_rounds: int = 2
    ema_rho: float = 0.70
    residual_small: float = 0.06
    residual_large: float = 0.35
    full_update_period: int = 4
    prediction_shrink: float = 0.75
    residual_topk_frac: float = 0.50

    # New parameters for aggregation-impact-aware QFL
    impact_small: float = 0.025
    impact_large: float = 0.080
    memory_shrink: float = 0.95


@dataclass
class FederatedConfig:
    rounds: int = 8
    seed: int = 42


@dataclass
class CommunicationCost:
    full_value_cost: float = 32.0
    residual_value_cost: float = 16.0
    metadata_cost: float = 64.0

    # Ebit accounting for teleportation-assisted correction
    teleport_ebit_cost: float = 1.0

    # Simulated teleportation quality
    teleport_fidelity: float = 0.95
    teleport_noise_scale: float = 0.02

# =============================================================================
# 2. Utility functions
# =============================================================================

def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap QNN angle parameters to [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def detect_feature_cols(df: pd.DataFrame, prefix: str = "x") -> List[str]:
    """Find feature columns x1, x2, ... and sort them numerically."""
    cols = []
    for c in df.columns:
        cs = str(c)
        if cs.startswith(prefix):
            suffix = cs[len(prefix):]
            if suffix.isdigit():
                cols.append(cs)
    cols = sorted(cols, key=lambda c: int(c[len(prefix):]))
    if not cols:
        raise ValueError(f"No feature columns found using prefix '{prefix}'. Available columns: {list(df.columns)}")
    return cols

def stratified_subset_by_class_and_client(
    df,
    size,
    target_col="target",
    client_col="client_id",
    seed=FINAL_SUBSET_SEED,
    stratify_by_client=True,
):
    """Create a deterministic QNN subset.

    If ``stratify_by_client`` is True, preserve target + existing client_id
    structure (used when retaining the original client partition). If synthetic
    Dirichlet clients will be created later, set it False so legacy client_id
    cannot influence row selection; the subset then preserves target only.
    """
    if size is None or size <= 0 or size >= len(df):
        return df.reset_index(drop=True)

    stratify_values = df[target_col]
    if stratify_by_client and client_col in df.columns:
        strat_col = (
            df[target_col].astype(str)
            + "_client_"
            + df[client_col].astype(str)
        )
        if strat_col.value_counts().min() >= 2:
            stratify_values = strat_col

    subset, _ = train_test_split(
        df,
        train_size=size,
        random_state=int(seed),
        stratify=stratify_values,
    )
    return subset.reset_index(drop=True)
# =============================================================================
# 3. Load processed  data
# =============================================================================

def load_processed_Vehical_Data(cfg: DataConfig):
    """
    Load your processed train/test files.

    Expected train/test columns:
        x1, x2, x3, x4, target, client_id

    If values_already_angles=False, x1..x4 are rescaled to [-pi, pi].
    This is recommended unless you explicitly saved them as angle features.
    """

    #train_df = pd.read_csv(cfg.train_csv)
    #test_df = pd.read_csv(cfg.test_csv)

    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.test_csv)

    # Reduce dataset size for fast QNN/QFL experiments
    train_df = stratified_subset_by_class_and_client(
        train_df,
        size=cfg.train_subset_size,
        target_col=cfg.target_col,
        client_col=cfg.client_col,
        seed=cfg.subset_seed,
        stratify_by_client=cfg.stratify_by_client,
    )

    test_df = stratified_subset_by_class_and_client(
        test_df,
        size=cfg.test_subset_size,
        target_col=cfg.target_col,
        client_col=cfg.client_col,
        seed=cfg.subset_seed,
        stratify_by_client=cfg.stratify_by_client,
    )

    feature_cols = detect_feature_cols(train_df, cfg.feature_prefix)
    feature_cols = feature_cols[:cfg.n_qubits]

    required = feature_cols + [cfg.target_col]
    if cfg.client_col not in train_df.columns:
        raise ValueError(f"Train file must contain '{cfg.client_col}' for federated client splits.")
    for col in required:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train file.")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test file.")

    X_train_raw = train_df[feature_cols].astype(float).to_numpy()
    X_test_raw = test_df[feature_cols].astype(float).to_numpy()

    if cfg.values_already_angles:
        X_train = np.clip(X_train_raw, -np.pi, np.pi)
        X_test = np.clip(X_test_raw, -np.pi, np.pi)
    else:
        scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        X_train = np.clip(X_train, -np.pi, np.pi)
        X_test = np.clip(X_test, -np.pi, np.pi)

    y_train = train_df[cfg.target_col].astype(int).to_numpy()
    y_test = test_df[cfg.target_col].astype(int).to_numpy()

    clients = []
    client_ids = sorted(train_df[cfg.client_col].unique())
    for cid in client_ids:
        sub = train_df[train_df[cfg.client_col] == cid]
        Xc_raw = sub[feature_cols].astype(float).to_numpy()
        if cfg.values_already_angles:
            Xc = np.clip(Xc_raw, -np.pi, np.pi)
        else:
            Xc = scaler.transform(Xc_raw)
            Xc = np.clip(Xc, -np.pi, np.pi)
        yc = sub[cfg.target_col].astype(int).to_numpy()
        
        try:
            cid_norm = int(cid)
        except (TypeError, ValueError):
            cid_norm = str(cid)
        clients.append((cid_norm, Xc, yc))

    info = {
        "feature_cols": feature_cols,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "target_distribution_train": train_df[cfg.target_col].value_counts().to_dict(),
        "target_distribution_test": test_df[cfg.target_col].value_counts().to_dict(),
        "client_sizes": {str(cid): int(len(yc)) for cid, _, yc in clients},
        "client_pos_rates": {str(cid): float(np.mean(yc)) for cid, _, yc in clients},
    }

    return X_train, X_test, y_train, y_test, clients, info

# =============================================================================
# 4. Compact statevector QNN
# =============================================================================

def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def apply_one_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    out = np.zeros_like(state)
    for basis_index, amp in enumerate(state):
        bits = list(format(basis_index, f"0{n_qubits}b"))
        if bits[control] == "1":
            bits[target] = "0" if bits[target] == "1" else "1"
        out[int("".join(bits), 2)] += amp
    return out


class StatevectorQNN:
    """
    Small binary QNN classifier.

    Input:
        x = [x1, x2, x3, x4] angle-encoded to n_qubits.

    Circuit:
        Data encoding: Ry(x_q), Rz(0.5*x_q^2)
        Ansatz: trainable Ry/Rz rotations + nearest-neighbour CNOT entanglement.

    Output:
        Probability of class 1 from measurement expectation of Z on qubit 0.
    """

    def __init__(self, cfg: QNNConfig):
        self.cfg = cfg
        self.n = cfg.n_qubits
        self.dim = 2 ** self.n

    def forward_one(self, x: np.ndarray, theta: np.ndarray) -> float:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        # Data encoding
        for q in range(self.n):
            state = apply_one_qubit_gate(state, ry(float(x[q])), q, self.n)
            state = apply_one_qubit_gate(state, rz(float(0.5 * x[q] * x[q])), q, self.n)

        # Variational ansatz
        p = 0
        for _ in range(self.cfg.ansatz_layers):
            for q in range(self.n):
                state = apply_one_qubit_gate(state, ry(float(theta[p])), q, self.n)
                p += 1
                state = apply_one_qubit_gate(state, rz(float(theta[p])), q, self.n)
                p += 1

            for q in range(self.n - 1):
                state = apply_cnot(state, q, q + 1, self.n)
            if self.n > 2:
                state = apply_cnot(state, self.n - 1, 0, self.n)

        # Measure Z on qubit 0 and map to class probability.
        probs = np.abs(state) ** 2
        exp_z = 0.0
        for basis_index, prob in enumerate(probs):
            bit0 = format(basis_index, f"0{self.n}b")[0]
            exp_z += (1.0 if bit0 == "0" else -1.0) * prob

        p1 = (1.0 - exp_z) / 2.0
        return float(np.clip(p1, 1e-7, 1.0 - 1e-7))

    def predict_proba(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        p1 = np.array([self.forward_one(x, theta) for x in X], dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X, theta)[:, 1] >= 0.5).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        p = self.predict_proba(X, theta)[:, 1]
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# =============================================================================
# 5. Explicit SPSA local optimizer
# =============================================================================

def spsa_train(
    qnn: StatevectorQNN,
    X: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    cfg: SPSAConfig,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Local QNN training with explicit SPSA hyperparameters.

    This avoids relying on default SPSA calibration, which can be unstable in
    small non-IID QFL clients.
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    n_params = len(theta)
    losses = []

    for k in range(cfg.local_iter):
        lr_k = cfg.learning_rate / ((k + 1) ** cfg.lr_decay)
        pert_k = cfg.perturbation / ((k + 1) ** cfg.pert_decay)

        if len(y) > cfg.batch_size:
            idx = rng.choice(len(y), size=cfg.batch_size, replace=False)
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y

        delta = rng.choice([-1.0, 1.0], size=n_params)

        loss_plus = qnn.loss(Xb, yb, wrap_pi(theta + pert_k * delta))
        loss_minus = qnn.loss(Xb, yb, wrap_pi(theta - pert_k * delta))
        grad_hat = ((loss_plus - loss_minus) / (2.0 * pert_k)) * delta

        step = -lr_k * grad_hat
        step_norm = np.linalg.norm(step)
        if step_norm > cfg.max_step_norm:
            step = step * (cfg.max_step_norm / (step_norm + 1e-12))

        theta = wrap_pi(theta + step)
        losses.append(qnn.loss(Xb, yb, theta))

    return theta, {
        "loss_first": float(losses[0]) if losses else np.nan,
        "loss_last": float(losses[-1]) if losses else np.nan,
        "loss_delta": float(losses[-1] - losses[0]) if len(losses) > 1 else 0.0,
    }


# =============================================================================
# 6. Prediction, residual communication, and aggregation
# =============================================================================
def _legacy_teleportation_proxy_DO_NOT_USE(
    update_vec,
    fidelity=0.95,
    noise_scale=0.02,
    rng=None
):
    """
    Simulated teleportation-assisted correction for high-impact QNN updates.

    This does not implement a physical Bell-state teleportation circuit.
    It models the effect of teleportation-assisted correction using:
        - ebit consumption proportional to transmitted parameters,
        - fidelity-dependent reconstruction noise,
        - wrapped QNN angle parameters.

    fidelity close to 1.0 means near-perfect reconstruction.
    lower fidelity means noisier reconstructed update.
    """
    if rng is None:
        rng = np.random.default_rng()

    update_vec = np.asarray(update_vec, dtype=float)
    n_params = len(update_vec)

    # Fidelity-dependent noise. Poorer fidelity gives larger distortion.
    sigma = noise_scale * (1.0 - fidelity)

    noise = rng.normal(loc=0.0, scale=sigma, size=n_params)

    reconstructed = wrap_pi(update_vec + noise)

    ebits_used = n_params

    return reconstructed, ebits_used, fidelity

class UpdatePredictor:
    """
    Client-wise exponential moving-average update predictor.

    The server uses previous client updates to predict the next client update.
    This is the simplest robust predictor and is enough for quick experiments.
    """

    def __init__(self, client_ids: List[int], n_params: int, rho: float):
        self.rho = rho
        self.pred = {cid: np.zeros(n_params, dtype=float) for cid in client_ids}
        self.ready = {cid: False for cid in client_ids}

    def predict(self, cid: int) -> np.ndarray:
        return self.pred[cid].copy()

    def update(self, cid: int, actual_update: np.ndarray) -> None:
        if not self.ready[cid]:
            self.pred[cid] = actual_update.copy()
            self.ready[cid] = True
        else:
            self.pred[cid] = wrap_pi(self.rho * actual_update + (1.0 - self.rho) * self.pred[cid])


def sparsify_topk(vec: np.ndarray, frac: float) -> Tuple[np.ndarray, int]:
    """Send only the largest residual coordinates."""
    if frac >= 1.0:
        return vec.copy(), len(vec)
    k = max(1, int(math.ceil(frac * len(vec))))
    idx = np.argsort(np.abs(vec))[-k:]
    out = np.zeros_like(vec)
    out[idx] = vec[idx]
    return out, k


def telesync_transport_update(
    update_vec: np.ndarray,
    seed: int = 12345,
):
    """
    Genuine TeleSync-style quantum transport of a client update using Qiskit/Aer.

    IMPORTANT:
    - This is an explicit Bell-pair teleportation circuit, not the previous
      fidelity/noise proxy.
    - A real-valued d-dimensional update is amplitude encoded into
      q = ceil(log2(d)) qubits.
    - q Bell pairs are generated.
    - Each source qubit is teleported with a Bell measurement plus X/Z
      feed-forward correction.
    - Aer returns the receiver density matrix so that the simulator can verify
      the received state and reconstruct the update for the surrounding
      classical QFL experiment.

    Resource accounting:
        ebits_used             = q
        correction_bits       = 2q
        side_information_bits = 33
            32 bits: update L2 norm
             1 bit : global-sign convention

    The exact receiver-state extraction used here is a simulator capability.
    A physical implementation that needs the full classical update vector
    would require tomography / repeated measurements. Therefore this method
    should be reported as a TeleSync-style quantum-resource baseline rather
    than as a classical bit-compression claim.
    """
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise ImportError(
            "telesync_style_qfl requires Qiskit and Qiskit Aer. "
            "Install with: python -m pip install qiskit qiskit-aer"
        ) from exc

    update = np.asarray(update_vec, dtype=float).reshape(-1)
    d = int(update.size)
    if d <= 0:
        raise ValueError("TeleSync update vector must be non-empty.")
    if not np.all(np.isfinite(update)):
        raise ValueError("TeleSync update contains non-finite values.")

    norm = float(np.linalg.norm(update))
    if norm <= 1e-15:
        return update.copy(), {
            "telesync_qubits": 0,
            "telesync_ebits": 0,
            "telesync_correction_bits": 0,
            "telesync_side_bits": 33,
            "telesync_fidelity": 1.0,
        }

    # Canonical global-sign convention. Global phase is physically irrelevant
    # but the classical update direction is not, so one side-information bit
    # preserves the original sign convention.
    nz = np.flatnonzero(np.abs(update) > 1e-15)
    first = int(nz[0])
    sign_bit = 1 if update[first] < 0 else 0
    canonical = (-update if sign_bit else update) / norm

    q = max(1, int(math.ceil(math.log2(d))))
    dim = 2 ** q
    amplitudes = np.zeros(dim, dtype=complex)
    amplitudes[:d] = canonical.astype(complex)
    amplitudes /= np.linalg.norm(amplitudes)

    src = QuantumRegister(q, "src")
    bell = QuantumRegister(q, "bell")
    recv = QuantumRegister(q, "recv")
    c_src = ClassicalRegister(q, "m_src")
    c_bell = ClassicalRegister(q, "m_bell")
    qc = QuantumCircuit(src, bell, recv, c_src, c_bell)

    # Prepare the parameter/update state on the sender register.
    qc.initialize(amplitudes, src)

    # Generate q EPR pairs.
    for i in range(q):
        qc.h(bell[i])
        qc.cx(bell[i], recv[i])

    # Standard teleportation for every encoded qubit.
    for i in range(q):
        qc.cx(src[i], bell[i])
        qc.h(src[i])
        qc.measure(src[i], c_src[i])
        qc.measure(bell[i], c_bell[i])

        # Dynamic classical feed-forward.
        with qc.if_test((c_bell[i], True)):
            qc.x(recv[i])
        with qc.if_test((c_src[i], True)):
            qc.z(recv[i])

    # Save only the receiver subsystem after feed-forward corrections.
    qc.save_density_matrix(recv, label="receiver")

    backend = AerSimulator(method="density_matrix")
    tqc = transpile(qc, backend)
    result = backend.run(
        tqc,
        shots=1,
        seed_simulator=int(seed),
    ).result()

    rho = np.asarray(result.data(0)["receiver"], dtype=complex)

    # Fidelity to the intended amplitude-encoded update state.
    fidelity = float(
        np.real(np.vdot(amplitudes, rho @ amplitudes))
    )
    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    # Simulator-only receiver-state recovery for integration with the existing
    # classical aggregation code.
    vals, vecs = np.linalg.eigh(rho)
    received = vecs[:, int(np.argmax(vals))]

    # Remove arbitrary global phase using a deterministic local convention.
    nz_recv = np.flatnonzero(np.abs(received) > 1e-12)
    if len(nz_recv):
        phase = np.angle(received[int(nz_recv[0])])
        received = received * np.exp(-1j * phase)
    received = np.real_if_close(received, tol=1000).real
    if received[int(nz_recv[0])] < 0 if len(nz_recv) else False:
        received = -received

    received /= max(float(np.linalg.norm(received)), 1e-15)
    reconstructed = received[:d] * norm
    if sign_bit:
        reconstructed = -reconstructed
    reconstructed = wrap_pi(np.asarray(reconstructed, dtype=float))

    resources = {
        "telesync_qubits": int(q),
        "telesync_ebits": int(q),
        "telesync_correction_bits": int(2 * q),
        "telesync_side_bits": 33,
        "telesync_fidelity": fidelity,
    }
    return reconstructed, resources


def evaluate(
    qnn: StatevectorQNN,
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:

    pred = qnn.predict(X, theta)
    proba = qnn.predict_proba(X, theta)[:, 1]

    unique_y = np.unique(y)

    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    # Balanced accuracy and AUC require both target classes
    # for meaningful binary evaluation.
    if len(unique_y) == 2:
        out["balanced_accuracy"] = float(
            balanced_accuracy_score(y, pred)
        )
        try:
            out["auc"] = float(
                roc_auc_score(y, proba)
            )
        except Exception:
            out["auc"] = np.nan
    else:
        out["balanced_accuracy"] = np.nan
        out["auc"] = np.nan

    return out

def aggregate(global_theta: np.ndarray, updates: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    avg_update = np.sum(weights[:, None] * np.vstack(updates), axis=0)
    return wrap_pi(global_theta + avg_update)



#=======================================================
import os
import json
import copy
import hashlib
import numpy as np
from dataclasses import dataclass, replace
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ============================================================
# FSB-UP context and S0-S3 protection profiles
# ============================================================

ALL_CONTEXT_FIELDS = frozenset({
    "sid", "round_id", "client_id", "direction", "mode", "task",
    "aggregation", "model_hash", "comm_method",
})

@dataclass(frozen=True)
class FSBContext:
    sid: str
    round_id: int
    client_id: str
    direction: str
    mode: str
    task: str
    aggregation: str
    model_hash: str
    comm_method: str

    def as_dict(self, excluded_fields=frozenset()):
        excluded_fields = frozenset(excluded_fields or ())
        obj = {
            "sid": self.sid,
            "round_id": int(self.round_id),
            "client_id": str(self.client_id),
            "direction": str(self.direction),
            "mode": str(self.mode),
            "task": str(self.task),
            "aggregation": str(self.aggregation),
            "model_hash": str(self.model_hash),
            "comm_method": str(self.comm_method),
        }
        return {k: v for k, v in obj.items() if k not in excluded_fields}

    def to_bytes(self, excluded_fields=frozenset()) -> bytes:
        return json.dumps(
            self.as_dict(excluded_fields),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def digest(self, excluded_fields=frozenset()) -> str:
        return hashlib.sha256(self.to_bytes(excluded_fields)).hexdigest()


def hash_array(x) -> str:
    arr = np.asarray(x, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def make_context_aad(
    ctx: FSBContext,
    seq: int,
    msg_type: str,
    excluded_fields=frozenset(),
) -> bytes:
    """Canonical FL/QFL context authenticated as AEAD associated data."""
    aad = {
        "profile": "FSB-UP-v1",
        "ctx": ctx.as_dict(excluded_fields),
        "seq": int(seq),
        "msg_type": str(msg_type),
    }
    return json.dumps(aad, sort_keys=True, separators=(",", ":")).encode()


# ============================================================
# Secret managers: ML-KEM establishes ss_i; FSB-UP does not replace KEM
# ============================================================

class PrototypeSecretManager:
    """Simulated shared secret for debugging only."""
    def __init__(self, client_ids):
        self.secrets = {str(cid): os.urandom(32) for cid in client_ids}

    def get_shared_secret(self, client_id):
        return self.secrets[str(client_id)]

    def get_kem_ciphertext_size(self, client_id):
        return 0

    def label(self):
        return "prototype"


class PurePythonMLKEMSecretManager:
    """Pure-Python ML-KEM manager using kyber-py."""
    def __init__(self, client_ids, algorithm="ML-KEM-768"):
        from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024

        if algorithm == "ML-KEM-512":
            self.kem = ML_KEM_512
        elif algorithm == "ML-KEM-768":
            self.kem = ML_KEM_768
        elif algorithm == "ML-KEM-1024":
            self.kem = ML_KEM_1024
        else:
            raise ValueError("algorithm must be ML-KEM-512, ML-KEM-768, or ML-KEM-1024")

        self.algorithm = algorithm
        self.shared = {}
        self.ciphertexts = {}
        self.public_keys = {}

        for cid in client_ids:
            cid = str(cid)
            ek, dk = self.kem.keygen()
            ss_server, ct = self.kem.encaps(ek)
            ss_client = self.kem.decaps(dk, ct)
            if ss_server != ss_client:
                raise ValueError(f"ML-KEM shared-secret mismatch for client {cid}")
            self.shared[cid] = ss_server
            self.ciphertexts[cid] = ct
            self.public_keys[cid] = ek

        print(f"[ML-KEM] Initialised pure-Python {algorithm} for {len(client_ids)} clients.")

    def get_shared_secret(self, client_id):
        return self.shared[str(client_id)]

    def get_kem_ciphertext_size(self, client_id):
        return len(self.ciphertexts[str(client_id)])

    def label(self):
        return f"pure_python_{self.algorithm}"


# ============================================================
# Replay cache: used only by S3 / Full FSB-UP
# ============================================================

class ReplayCache:
    def __init__(self):
        self.seen = set()

    def check_and_add(self, ctx_hash, seq, client_id, direction):
        key = (str(ctx_hash), int(seq), str(client_id), str(direction))
        if key in self.seen:
            raise ValueError("REPLAY_REJECT")
        self.seen.add(key)


# ============================================================
# Key derivation
# ============================================================

def derive_traffic_key(shared_secret: bytes) -> bytes:
    """S0/S1 traffic key: no FL/QFL semantic context in the KDF."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"VEHICULAR-QFL-TRAFFIC-v1",
    )
    return hkdf.derive(shared_secret)


def derive_context_key(
    shared_secret: bytes,
    ctx: FSBContext,
    excluded_fields=frozenset(),
) -> bytes:
    """S2/S3 key: K_i,t = HKDF(ss_i, 'FSB-UP-v1' || H(ctx_i,t))."""
    ctx_hash = bytes.fromhex(ctx.digest(excluded_fields))
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"FSB-UP-v1|" + ctx_hash,
    )
    return hkdf.derive(shared_secret)


# ============================================================
# Packet and update validation
# ============================================================

@dataclass
class SecurePacket:
    client_id: str
    round_id: int
    direction: str
    ctx_hash: str
    seq: int
    nonce: bytes
    ciphertext: bytes
    packet_bytes: int


def serialize_update(update_vec) -> bytes:
    arr = np.asarray(update_vec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("UPDATE_ENCODE_REJECT")
    return arr.tobytes()


def deserialize_and_validate_update(raw: bytes, expected_shape):
    expected_count = int(np.prod(expected_shape))
    expected_bytes = expected_count * np.dtype(np.float64).itemsize
    if len(raw) != expected_bytes:
        raise ValueError("UPDATE_DECODE_REJECT")
    arr = np.frombuffer(raw, dtype=np.float64).copy().reshape(expected_shape)
    if not np.all(np.isfinite(arr)):
        raise ValueError("UPDATE_DECODE_REJECT")
    return arr


SECURITY_PROFILES = {
    "S0_MLKEM_AEAD": "s0",
    "S1_MLKEM_CONTEXT_AAD": "s1",
    "S2_MLKEM_CONTEXT_KEY": "s2",
    "S3_FSB_UP": "s3",
    # Legacy aliases kept only so old experiment cells do not break.
    "mlkem_aead_only": "s0",
    "mlkem_fsb_ake": "s3",
    "mlkem_fsb_up": "s3",
}


def normalize_security_profile(security_mode):
    if security_mode not in SECURITY_PROFILES:
        raise ValueError(
            f"Unknown security_mode={security_mode}. "
            f"Choose one of {list(SECURITY_PROFILES)}"
        )
    return SECURITY_PROFILES[security_mode]


def protect_update(
    update_vec,
    shared_secret,
    ctx,
    seq,
    msg_type="vehicular_qfl_update",
    security_profile="s3",
    excluded_context_fields=frozenset(),
):
    excluded_context_fields = frozenset(excluded_context_fields or ())

    if security_profile == "s0":
        key = derive_traffic_key(shared_secret)
        aad = b"VEHICULAR-QFL-AEAD-v1"
        ctx_hash = "unbound"

    elif security_profile == "s1":
        key = derive_traffic_key(shared_secret)
        aad = make_context_aad(ctx, seq, msg_type, excluded_context_fields)
        ctx_hash = "aad-bound"

    elif security_profile == "s2":
        key = derive_context_key(shared_secret, ctx, excluded_context_fields)
        aad = b"VEHICULAR-QFL-S2-v1"
        ctx_hash = "key-bound"

    elif security_profile == "s3":
        key = derive_context_key(shared_secret, ctx, excluded_context_fields)
        aad = make_context_aad(ctx, seq, msg_type, excluded_context_fields)
        ctx_hash = ctx.digest(excluded_context_fields)

    else:
        raise ValueError(f"Unknown security_profile={security_profile}")

    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    plaintext = serialize_update(update_vec)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

    packet_bytes = (
        len(str(ctx_hash).encode())
        + len(str(seq).encode())
        + len(nonce)
        + len(ciphertext)
    )

    return SecurePacket(
        client_id=str(ctx.client_id),
        round_id=int(ctx.round_id),
        direction=str(ctx.direction),
        ctx_hash=ctx_hash,
        seq=int(seq),
        nonce=nonce,
        ciphertext=ciphertext,
        packet_bytes=packet_bytes,
    )


def _s3_expected_state_check(packet, expected_ctx, excluded_context_fields=frozenset()):
    """Explicit receiver-side expected-state checks used only by full FSB-UP."""
    excluded_context_fields = frozenset(excluded_context_fields or ())

    # Direct packet metadata checks. If a field is deliberately ablated, its
    # corresponding direct check is also removed so the ablation is meaningful.
    if "client_id" not in excluded_context_fields and str(packet.client_id) != str(expected_ctx.client_id):
        raise ValueError("CONTEXT_REJECT")
    if "round_id" not in excluded_context_fields and int(packet.round_id) != int(expected_ctx.round_id):
        raise ValueError("CONTEXT_REJECT")
    if "direction" not in excluded_context_fields and str(packet.direction) != str(expected_ctx.direction):
        raise ValueError("CONTEXT_REJECT")

    if packet.ctx_hash != expected_ctx.digest(excluded_context_fields):
        raise ValueError("CONTEXT_REJECT")


def open_update(
    packet,
    shared_secret,
    expected_ctx,
    replay_cache,
    expected_shape,
    msg_type="vehicular_qfl_update",
    security_profile="s3",
    excluded_context_fields=frozenset(),
):
    excluded_context_fields = frozenset(excluded_context_fields or ())

    if security_profile == "s0":
        key = derive_traffic_key(shared_secret)
        aad = b"VEHICULAR-QFL-AEAD-v1"

    elif security_profile == "s1":
        key = derive_traffic_key(shared_secret)
        aad = make_context_aad(expected_ctx, packet.seq, msg_type, excluded_context_fields)

    elif security_profile == "s2":
        key = derive_context_key(shared_secret, expected_ctx, excluded_context_fields)
        aad = b"VEHICULAR-QFL-S2-v1"

    elif security_profile == "s3":
        _s3_expected_state_check(packet, expected_ctx, excluded_context_fields)
        key = derive_context_key(shared_secret, expected_ctx, excluded_context_fields)
        aad = make_context_aad(expected_ctx, packet.seq, msg_type, excluded_context_fields)

    else:
        raise ValueError(f"Unknown security_profile={security_profile}")

    # Authenticate first. Do not poison the replay cache with unauthenticated data.
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(packet.nonce, packet.ciphertext, aad)
    update = deserialize_and_validate_update(plaintext, expected_shape)

    if security_profile == "s3":
        replay_cache.check_and_add(
            expected_ctx.digest(excluded_context_fields),
            packet.seq,
            packet.client_id,
            packet.direction,
        )

    return update


# ============================================================
# Threat models
# ============================================================

def tamper_ciphertext(packet):
    pkt = copy.deepcopy(packet)
    ct = bytearray(pkt.ciphertext)
    if len(ct) > 0:
        ct[0] ^= 1
    pkt.ciphertext = bytes(ct)
    return pkt


def build_attack_context(ctx, attack_mode):
    if attack_mode == "wrong_round":
        return replace(ctx, round_id=ctx.round_id + 1)
    if attack_mode == "client_swap":
        return replace(ctx, client_id=f"{ctx.client_id}_swapped")
    if attack_mode == "direction_confusion":
        return replace(ctx, direction="downlink" if ctx.direction == "uplink" else "uplink")
    if attack_mode == "mode_downgrade":
        return replace(ctx, mode="downgraded_mode")
    if attack_mode == "task_substitution":
        return replace(ctx, task="DifferentVehicularTask")
    if attack_mode == "aggregation_swap":
        return replace(ctx, aggregation="A2G" if ctx.aggregation == "FedAvg" else "FedAvg")
    if attack_mode == "stale_model":
        return replace(ctx, model_hash="stale_global_model_hash")
    if attack_mode in {"comm_method_swap", "communication_mode_swap"}:
        return replace(ctx, comm_method="different_comm_method")
    return ctx


def classify_security_error(e):
    msg = str(e)
    if "REPLAY_REJECT" in msg:
        return "replay_reject"
    if "CONTEXT_REJECT" in msg:
        return "context_reject"
    if "UPDATE_DECODE_REJECT" in msg or "UPDATE_ENCODE_REJECT" in msg:
        return "update_validation_reject"
    return "aead_or_other_reject"

#===========================================================
#Add nOISE HELPER FUNCTIONS
#==============================================================

def flip_random_byte_in_ciphertext(packet, rng):
    """
    Simulates random channel corruption or byte-level tampering.
    Expected result: AEAD rejection.
    """
    pkt = copy.deepcopy(packet)

    if not hasattr(pkt, "ciphertext") or len(pkt.ciphertext) == 0:
        return pkt

    ct = bytearray(pkt.ciphertext)
    idx = int(rng.integers(0, len(ct)))
    bit = 1 << int(rng.integers(0, 8))
    ct[idx] ^= bit
    pkt.ciphertext = bytes(ct)

    return pkt


def choose_random_semantic_attack(rng):
    semantic_attacks = [
        "wrong_round",
        "client_swap",
        "direction_confusion",
        "mode_downgrade",
        "stale_model",
        "task_substitution",
        "aggregation_swap",
        "comm_method_swap",
    ]
    return semantic_attacks[int(rng.integers(0, len(semantic_attacks)))]


def apply_depolarizing_update_proxy(update_vec, qnoise_cfg, rng):
    """
    Depolarizing-inspired update-level proxy.

    This does not replace a full noisy quantum simulator.
    It approximates the effect of quantum noise by shrinking useful update
    information and adding bounded random perturbation.

    Use this when your current StatevectorQNN implementation is noiseless.
    """
    update_vec = np.asarray(update_vec, dtype=float)

    if qnoise_cfg is None or qnoise_cfg.depolarizing_p <= 0.0:
        return update_vec

    p = float(qnoise_cfg.depolarizing_p)

    # Depolarization reduces useful signal.
    shrink = max(0.0, 1.0 - p)

    # Add small angular perturbation proportional to sqrt(p).
    sigma = qnoise_cfg.update_noise_scale * np.sqrt(p)
    perturb = rng.normal(loc=0.0, scale=sigma, size=update_vec.shape)

    noisy_update = shrink * update_vec + perturb

    return wrap_pi(noisy_update)


def teleport_fidelity_from_depolarizing(qnoise_cfg):
    """
    Converts depolarizing probability into a simple teleportation fidelity proxy.
    """
    if qnoise_cfg is None:
        return None

    p = float(qnoise_cfg.depolarizing_p)

    # Simple monotonic proxy: fidelity decreases as p increases.
    return max(float(qnoise_cfg.teleport_fidelity_floor), 1.0 - p)


#====================================================
#Add noise configuration classes
#===================================================


from dataclasses import dataclass
import copy
import numpy as np

@dataclass
class ChannelNoiseConfig:
    """
    Classical vehicular-channel noise.
    This affects packet delivery and secure transport.
    """
    packet_loss_prob: float = 0.0
    bit_flip_prob: float = 0.0
    duplicate_prob: float = 0.0
    delay_prob: float = 0.0
    context_drift_prob: float = 0.0
    seed: int = 123


@dataclass
class QuantumNoiseConfig:
    """
    Quantum/QFL-side noise.

    For aggregation_impact_qfl:
        Uses depolarizing-inspired update perturbation.

    telesync_style_qfl uses an explicit Qiskit/Aer Bell-pair circuit and does not use the old fidelity proxy.
    """
    depolarizing_p: float = 0.0
    update_noise_scale: float = 0.05
    teleport_fidelity_floor: float = 0.50
    seed: int = 456


#----Run security qfl-----
def run_qfl_secure_comm(
    method: str,
    qnn,
    clients,
    X_test,
    y_test,
    fed_cfg,
    spsa_cfg,
    pred_cfg,
    comm_cfg,
    security_mode="S3_FSB_UP",
    secret_source="mlkem_python",
    mlkem_algorithm="ML-KEM-768",
    attack_mode="none",
    attack_client_id=None,
    task_name="HDSVT",
    excluded_context_fields=frozenset(),

    # New experiment-2 and experiment-3 controls
    channel_noise_cfg=None,
    channel_noise_name="clean",
    quantum_noise_cfg=None,
    quantum_noise_name="ideal",


):
    """
    Secure vehicular QFL communication runner.

    method:
        tuned_qfl, predictive_residual_qfl, aggregation_impact_qfl, telesync_style_qfl

    security_mode:
        S0_MLKEM_AEAD         -> ordinary AEAD, no FL context binding
        S1_MLKEM_CONTEXT_AAD  -> context in AEAD associated data
        S2_MLKEM_CONTEXT_KEY  -> context-derived key, no full receiver pipeline
        S3_FSB_UP             -> full FSB-UP profile

    secret_source:
        prototype        -> simulated 32-byte shared secret
        mlkem_python     -> pure-Python ML-KEM-768 via kyber-py
    """

    assert method in {
        "tuned_qfl",
        "predictive_residual_qfl",
        "aggregation_impact_qfl",
        "telesync_style_qfl",
    }

    security_profile = normalize_security_profile(security_mode)
    excluded_context_fields = frozenset(excluded_context_fields or ())

    rng = np.random.default_rng(fed_cfg.seed)
    client_ids = [cid for cid, _, _ in clients]
    n_params = qnn.cfg.n_params


    channel_rng = np.random.default_rng(
    getattr(channel_noise_cfg, "seed", fed_cfg.seed + 777)
        )

    quantum_rng = np.random.default_rng(
        getattr(quantum_noise_cfg, "seed", fed_cfg.seed + 888)
    )


    global_theta = rng.uniform(-0.05, 0.05, size=n_params)

    sizes = np.array([len(yc) for _, _, yc in clients], dtype=float)
    weights = sizes / sizes.sum()

    predictor = UpdatePredictor(client_ids, n_params, pred_cfg.ema_rho)

    client_update_memory = {
        cid: np.zeros(n_params, dtype=float)
        for cid in client_ids
    }

    memory_ready = {
        cid: False
        for cid in client_ids
    }

    # ------------------------------------------------------------------
    # Secret manager
    # ------------------------------------------------------------------
    if secret_source == "prototype":
        secret_manager = PrototypeSecretManager(client_ids)
    elif secret_source == "mlkem_python":
        secret_manager = PurePythonMLKEMSecretManager(client_ids, algorithm=mlkem_algorithm)
    else:
        raise ValueError("secret_source must be 'prototype' or 'mlkem_python'")

    replay_cache = ReplayCache()

    if attack_client_id is None:
        attack_client_id = client_ids[0]

    round_rows = []
    client_rows = []
    security_rows = []

    for t in range(1, fed_cfg.rounds + 1):

        reconstructed_updates = []
        # Keep the original FedAvg data-size weight for every update that is
        # actually accepted by the server.  If packets are dropped/rejected,
        # the surviving weights are renormalised over the accepted set.
        accepted_weights = []
        # Communication accounting is separated into algorithmic payload cost
        # and cryptographic/security packet cost. Legacy ``bits`` fields below
        # remain wire-bit aliases for backward compatibility with security plots.
        round_algorithmic_bits = 0.0
        round_security_bits = 0.0
        round_wire_bits = 0.0
        round_ebits_proxy = 0.0

        mode_counts = {
            "full": 0,
            "prediction_only": 0,
            "residual": 0,
            "full_fallback": 0,
            "memory_reuse": 0,
            "impact_residual": 0,
            "telesync_state_sync": 0,
            "security_reject": 0,
        }

        residual_norms = []
        impact_scores = []

        total_packets = 0
        accepted_packets = 0
        attack_attempts = 0
        attack_successes = 0
        context_rejects = 0
        replay_rejects = 0
        aead_or_other_rejects = 0
        secure_packet_bytes = []

        h_G = hash_array(global_theta)

        for local_idx, (cid, Xc, yc) in enumerate(clients):

            local_theta, train_info = spsa_train(
                qnn,
                Xc,
                yc,
                global_theta,
                spsa_cfg,
                seed=fed_cfg.seed + 1000 * t + local_idx,
            )

            actual_update_clean = wrap_pi(local_theta - global_theta)

            # Experiment 3: quantum/depolarizing-inspired update noise
            #UNDER under quantum_noise_name="ideal", no change; under depolarizing noise, the local update becomes noisier;
            actual_update = apply_depolarizing_update_proxy(
                actual_update_clean,
                quantum_noise_cfg,
                quantum_rng
            )

            # ============================================================
            # Original communication decision
            # ============================================================

            if method == "tuned_qfl":
                reconstructed_update = actual_update.copy()
                mode = "full"
                sent_values = n_params
                bits = n_params * comm_cfg.full_value_cost
                ebits_proxy = 0.0
                residual_norm = np.nan
                impact_score = np.nan

            elif method == "predictive_residual_qfl":
                pred_update = predictor.predict(cid)
                residual = wrap_pi(actual_update - pred_update)
                residual_norm = float(np.linalg.norm(residual))
                residual_norms.append(residual_norm)
                impact_score = np.nan

                force_full = (t <= pred_cfg.warmup_rounds) or (
                    pred_cfg.full_update_period > 0
                    and t % pred_cfg.full_update_period == 0
                )

                if force_full:
                    reconstructed_update = actual_update.copy()
                    mode = "full"
                    sent_values = n_params
                    bits = n_params * comm_cfg.full_value_cost + comm_cfg.metadata_cost
                    ebits_proxy = 0.0

                elif residual_norm <= pred_cfg.residual_small:
                    reconstructed_update = pred_cfg.prediction_shrink * pred_update
                    mode = "prediction_only"
                    sent_values = 0
                    bits = comm_cfg.metadata_cost
                    ebits_proxy = 0.0

                elif residual_norm <= pred_cfg.residual_large:
                    # Actual transmitted representation: Top-K coordinates with
                    # 16-bit symmetric quantization, explicit coordinate-index
                    # bits, one 32-bit quantization scale, and metadata.
                    residual_to_send, residual_idx, _, _, residual_acc = _quantized_topk_residual(
                        residual,
                        frac=pred_cfg.residual_topk_frac,
                        bits=int(comm_cfg.residual_value_cost),
                        metadata_bits=int(comm_cfg.metadata_cost),
                    )
                    sent_values = int(len(residual_idx))
                    reconstructed_update = wrap_pi(pred_update + residual_to_send)
                    mode = "residual"
                    bits = float(residual_acc.total_bits)
                    ebits_proxy = 0.0

                else:
                    reconstructed_update = actual_update.copy()
                    mode = "full_fallback"
                    sent_values = n_params
                    bits = n_params * comm_cfg.full_value_cost + comm_cfg.metadata_cost
                    ebits_proxy = n_params * comm_cfg.teleport_ebit_cost

                predictor.update(cid, reconstructed_update)

            elif method == "aggregation_impact_qfl":
                previous_update = client_update_memory[cid]

                if memory_ready[cid]:
                    impact_vector = wrap_pi(actual_update - previous_update)
                    impact_score = float(weights[local_idx] * np.linalg.norm(impact_vector))
                else:
                    impact_vector = actual_update.copy()
                    impact_score = np.inf

                impact_scores.append(impact_score if np.isfinite(impact_score) else np.nan)
                residual_norm = float(np.linalg.norm(impact_vector))
                residual_norms.append(residual_norm)

                force_full = (
                    not memory_ready[cid]
                    or t <= pred_cfg.warmup_rounds
                    or (
                        pred_cfg.full_update_period > 0
                        and t % pred_cfg.full_update_period == 0
                    )
                )

                if force_full:
                    reconstructed_update = actual_update.copy()
                    mode = "full"
                    sent_values = n_params
                    bits = n_params * comm_cfg.full_value_cost + comm_cfg.metadata_cost
                    ebits_proxy = 0.0
                    client_update_memory[cid] = reconstructed_update.copy()
                    memory_ready[cid] = True

                elif impact_score <= pred_cfg.impact_small:
                    reconstructed_update = pred_cfg.memory_shrink * previous_update
                    mode = "memory_reuse"
                    sent_values = 0
                    bits = comm_cfg.metadata_cost
                    ebits_proxy = 0.0
                    client_update_memory[cid] = reconstructed_update.copy()

                elif impact_score <= pred_cfg.impact_large:
                    correction_to_send, correction_idx, _, _, correction_acc = _quantized_topk_residual(
                        impact_vector,
                        frac=pred_cfg.residual_topk_frac,
                        bits=int(comm_cfg.residual_value_cost),
                        metadata_bits=int(comm_cfg.metadata_cost),
                    )
                    sent_values = int(len(correction_idx))
                    reconstructed_update = wrap_pi(previous_update + correction_to_send)
                    mode = "impact_residual"
                    bits = float(correction_acc.total_bits)
                    ebits_proxy = 0.0
                    client_update_memory[cid] = reconstructed_update.copy()

                else:
                    reconstructed_update = actual_update.copy()
                    mode = "full_fallback"
                    sent_values = n_params
                    bits = n_params * comm_cfg.full_value_cost + comm_cfg.metadata_cost
                    ebits_proxy = 0.0
                    client_update_memory[cid] = reconstructed_update.copy()
                    memory_ready[cid] = True

            elif method == "telesync_style_qfl":
                reconstructed_update, tele = telesync_transport_update(
                    actual_update,
                    seed=fed_cfg.seed + 9000 * t + local_idx,
                )
                mode = "telesync_state_sync"
                sent_values = 0
                bits = (
                    tele["telesync_correction_bits"]
                    + tele["telesync_side_bits"]
                    + comm_cfg.metadata_cost
                )
                ebits_proxy = tele["telesync_ebits"]
                residual_norm = float(np.linalg.norm(
                    wrap_pi(actual_update - reconstructed_update)
                ))
                residual_norms.append(residual_norm)
                impact_score = np.nan

            # ============================================================
            # FSB-UP / S0-S3 communication protection boundary
            # ============================================================

            total_packets += 1

            ctx = FSBContext(
                sid="VEHICULAR_QFL_FSB_UP",
                round_id=t,
                client_id=str(cid),
                direction="uplink",
                mode=security_mode,
                task=task_name,
                aggregation="FedAvg",
                model_hash=h_G,
                comm_method=method,
            )

            shared_secret = secret_manager.get_shared_secret(cid)

            packet = protect_update(
                reconstructed_update,
                shared_secret,
                ctx,
                seq=t,
                msg_type="vehicular_qfl_update",
                security_profile=security_profile,
                excluded_context_fields=excluded_context_fields,
            )

            packet_to_verify = packet
            expected_ctx = ctx

            noise_reason = "none"
            packet_dropped = False
            duplicate_replay_noise = False

            # ============================================================
            # Experiment 2: classical vehicular channel noise
            # ============================================================

            if channel_noise_cfg is not None:

                # Packet loss: packet never reaches server
                if channel_rng.random() < channel_noise_cfg.packet_loss_prob:
                    packet_dropped = True
                    noise_reason = "packet_loss"

                # Bit corruption: ciphertext modified by channel noise
                elif channel_rng.random() < channel_noise_cfg.bit_flip_prob:
                    packet_to_verify = flip_random_byte_in_ciphertext(packet_to_verify, channel_rng)
                    noise_reason = "bit_flip_corruption"

                # Context drift: packet is interpreted under wrong semantic context
                elif channel_rng.random() < channel_noise_cfg.context_drift_prob:
                    random_sem_attack = choose_random_semantic_attack(channel_rng)
                    expected_ctx = build_attack_context(ctx, random_sem_attack)
                    noise_reason = f"context_drift_{random_sem_attack}"

                # Delay: packet effectively belongs to a wrong/stale round
                elif channel_rng.random() < channel_noise_cfg.delay_prob:
                    expected_ctx = build_attack_context(ctx, "wrong_round")
                    noise_reason = "delayed_wrong_round"

                # Duplicate: packet arrives twice
                elif channel_rng.random() < channel_noise_cfg.duplicate_prob:
                    duplicate_replay_noise = True
                    noise_reason = "duplicate_replay"

            kem_ct_bytes = secret_manager.get_kem_ciphertext_size(cid)
            packet_bytes = packet.packet_bytes + kem_ct_bytes
            secure_packet_bytes.append(packet_bytes)

            attacked = (attack_mode != "none" and cid == attack_client_id)

            # Do NOT reset packet_to_verify/expected_ctx here. They may already
            # contain channel-noise effects from the block above. The earlier
            # notebook reset them at this point, which unintentionally erased
            # context-drift/bit-corruption/delay injections before verification.
            if attacked:
                attack_attempts += 1

                if attack_mode in {"mitm_tamper", "ciphertext_tampering"}:
                    packet_to_verify = tamper_ciphertext(packet)

                elif attack_mode == "replay":
                    pass

                else:
                    expected_ctx = build_attack_context(ctx, attack_mode)

            uplink_ok = 1
            reject_reason = "none"


            if packet_dropped:
                uplink_ok = 0
                reject_reason = "packet_loss"
                mode_counts["security_reject"] += 1

                # Do not append this update to reconstructed_updates.
                # It is a missing packet, not a cryptographic rejection.
                client_rows.append({
                    "method": method,
                    "security_mode": security_mode,
                    "context_ablation": "+".join(sorted(excluded_context_fields)) if excluded_context_fields else "none",
                    "secret_source": secret_manager.label(),
                    "attack_mode": attack_mode,
                    "channel_noise_name": channel_noise_name,
                    "quantum_noise_name": quantum_noise_name,
                    "depolarizing_p": getattr(quantum_noise_cfg, "depolarizing_p", 0.0),
                    "round": t,
                    "client_id": cid,
                    "client_size": len(yc),
                    "fedavg_base_weight": float(weights[local_idx]),
                    "mode": mode,
                    "uplink_ok": int(uplink_ok),
                    "reject_reason": reject_reason,
                    "noise_reason": noise_reason,
                    "secure_packet_bytes": float(packet_bytes),
                    "update_norm": float(np.linalg.norm(actual_update)),
                    "clean_update_norm": float(np.linalg.norm(actual_update_clean)),

                    "channel_noise_name": channel_noise_name,
                    "quantum_noise_name": quantum_noise_name,
                    "depolarizing_p": getattr(quantum_noise_cfg, "depolarizing_p", 0.0),
                    "noise_reason": noise_reason,
                })

                continue

            try:
                opened_update = open_update(
                    packet_to_verify,
                    shared_secret,
                    expected_ctx,
                    replay_cache,
                    expected_shape=reconstructed_update.shape,
                    msg_type="vehicular_qfl_update",
                    security_profile=security_profile,
                excluded_context_fields=excluded_context_fields,
                )

                if attacked and attack_mode == "replay":
                    try:
                        _ = open_update(
                            packet_to_verify,
                            shared_secret,
                            expected_ctx,
                            replay_cache,
                            expected_shape=reconstructed_update.shape,
                            msg_type="vehicular_qfl_update",
                            security_profile=security_profile,
                excluded_context_fields=excluded_context_fields,
                        )
                        attack_successes += 1
                        reject_reason = "replay_not_rejected"
                    except Exception as e:
                        reject_reason = classify_security_error(e)
                        if reject_reason == "replay_reject":
                            replay_rejects += 1
                        elif reject_reason == "context_reject":
                            context_rejects += 1
                        else:
                            aead_or_other_rejects += 1

                elif attacked:
                    attack_successes += 1

                accepted_packets += 1
                reconstructed_updates.append(opened_update)
                accepted_weights.append(float(weights[local_idx]))

                # If duplicate packet noise occurs, submit the same packet again.
                # Expected result: replay rejection.
                if duplicate_replay_noise:
                    try:
                        _ = open_update(
                            packet_to_verify,
                            shared_secret,
                            expected_ctx,
                            replay_cache,
                            expected_shape=reconstructed_update.shape,
                            msg_type="vehicular_qfl_update",
                            security_profile=security_profile,
                excluded_context_fields=excluded_context_fields,
                        )

                        # If the duplicate is accepted, that is a failure.
                        attack_successes += 1
                        reject_reason = "duplicate_replay_accepted"

                    except Exception as e:
                        dup_reason = classify_security_error(e)

                        if dup_reason == "replay_reject":
                            replay_rejects += 1
                        elif dup_reason == "context_reject":
                            context_rejects += 1
                        else:
                            aead_or_other_rejects += 1

            except Exception as e:
                uplink_ok = 0
                mode_counts["security_reject"] += 1
                reject_reason = classify_security_error(e)

                if reject_reason == "context_reject":
                    context_rejects += 1
                elif reject_reason == "replay_reject":
                    replay_rejects += 1
                else:
                    aead_or_other_rejects += 1

                # Important: if rejected, do not aggregate this client.
                # Also adjust weight handling later using accepted updates only.

            security_bits = 8.0 * packet_bytes
            wire_bits = float(bits) + security_bits
            round_algorithmic_bits += float(bits)
            round_security_bits += security_bits
            round_wire_bits += wire_bits
            round_ebits_proxy += ebits_proxy

            if mode in mode_counts:
                mode_counts[mode] += 1

            local_eval = evaluate(qnn, local_theta, Xc, yc)

            client_rows.append({
                "method": method,
                "security_mode": security_mode,
                    "context_ablation": "+".join(sorted(excluded_context_fields)) if excluded_context_fields else "none",
                "secret_source": secret_manager.label(),
                "attack_mode": attack_mode,
                "round": t,
                "client_id": cid,
                "client_size": len(yc),
                "client_pos_rate": float(np.mean(yc)),
                "fedavg_base_weight": float(weights[local_idx]),
                "mode": mode,
                "uplink_ok": int(uplink_ok),
                "reject_reason": reject_reason,
                "sent_values": int(sent_values),
                # ``bits`` remains the logical/algorithmic method cost at client level.
                "bits": float(bits),
                "algorithmic_bits": float(bits),
                "security_bits": float(security_bits),
                "wire_bits": float(wire_bits),
                "secure_packet_bytes": float(packet_bytes),
                "ebits_proxy": float(ebits_proxy),
                "update_norm": float(np.linalg.norm(actual_update)),
                "residual_norm": residual_norm,
                "impact_score": impact_score,
                **{f"local_{k}": v for k, v in local_eval.items()},
                **train_info,
                "channel_noise_name": channel_noise_name,
                "quantum_noise_name": quantum_noise_name,
                "depolarizing_p": getattr(quantum_noise_cfg, "depolarizing_p", 0.0),
                "noise_reason": noise_reason,
            })

        if len(reconstructed_updates) == 0:
            raise RuntimeError(f"All client updates rejected at round {t}.")

        # Fair FedAvg aggregation for the proposed communication methods:
        # preserve each accepted client's original data-size weight and
        # renormalise only over the updates that actually reached/verified at
        # the server.  This keeps the communication strategy as the proposed
        # contribution while holding the underlying FL aggregation rule fixed.
        accepted_weights = np.asarray(accepted_weights, dtype=float)
        if len(accepted_weights) != len(reconstructed_updates):
            raise RuntimeError(
                f"Accepted-weight/update mismatch at round {t}: "
                f"{len(accepted_weights)} weights for "
                f"{len(reconstructed_updates)} updates."
            )

        accepted_weight_mass = float(accepted_weights.sum())
        if not np.isfinite(accepted_weight_mass) or accepted_weight_mass <= 0.0:
            raise RuntimeError(
                f"Invalid accepted FedAvg weight mass at round {t}: "
                f"{accepted_weight_mass}"
            )

        accepted_weights = accepted_weights / accepted_weight_mass
        global_theta = aggregate(
            global_theta,
            reconstructed_updates,
            accepted_weights,
        )

        test_eval = evaluate(qnn, global_theta, X_test, y_test)

        prev_wire_bits = round_rows[-1].get("cumulative_wire_bits", round_rows[-1].get("cumulative_bits", 0.0)) if round_rows else 0.0
        prev_algorithmic_bits = round_rows[-1].get("cumulative_algorithmic_bits", 0.0) if round_rows else 0.0
        prev_security_bits = round_rows[-1].get("cumulative_security_bits", 0.0) if round_rows else 0.0
        prev_ebits = round_rows[-1]["cumulative_ebits_proxy"] if round_rows else 0.0
        finite_impact_scores = [
            x for x in impact_scores
            if np.isfinite(x)
        ]

        mean_impact_score = (
            float(np.mean(finite_impact_scores))
            if finite_impact_scores
            else np.nan
        )
        round_rows.append({
            "method": method,
            "security_mode": security_mode,
                    "context_ablation": "+".join(sorted(excluded_context_fields)) if excluded_context_fields else "none",
            "secret_source": secret_manager.label(),
            "attack_mode": attack_mode,
            "round": t,
            # Legacy aliases retain the historical secure-wire meaning.
            "bits": float(round_wire_bits),
            "cumulative_bits": float(prev_wire_bits + round_wire_bits),
            # Use these explicit fields for the final communication-efficiency analysis.
            "algorithmic_bits": float(round_algorithmic_bits),
            "cumulative_algorithmic_bits": float(prev_algorithmic_bits + round_algorithmic_bits),
            "security_bits": float(round_security_bits),
            "cumulative_security_bits": float(prev_security_bits + round_security_bits),
            "wire_bits": float(round_wire_bits),
            "cumulative_wire_bits": float(prev_wire_bits + round_wire_bits),
            "ebits_proxy": float(round_ebits_proxy),
            "cumulative_ebits_proxy": float(prev_ebits + round_ebits_proxy),
            "mean_secure_packet_bytes": float(np.mean(secure_packet_bytes)),
            "accepted_packets": int(accepted_packets),
            "total_packets": int(total_packets),
            "uplink_accept_rate": float(accepted_packets / max(total_packets, 1)),
            "aggregation_weighting": "data_size_fedavg_renormalized_over_accepted",
            "accepted_weight_mass_before_renorm": float(accepted_weight_mass),
            "attack_attempts": int(attack_attempts),
            "attack_successes": int(attack_successes),
            "attack_accept_rate": float(attack_successes / max(attack_attempts, 1)) if attack_attempts > 0 else 0.0,
            "context_rejects": int(context_rejects),
            "replay_rejects": int(replay_rejects),
            "aead_or_other_rejects": int(aead_or_other_rejects),
            "mean_residual_norm": float(np.nanmean(residual_norms)) if len(residual_norms) else np.nan,
            "mean_impact_score": mean_impact_score,
            #"mean_impact_score": float(np.nanmean(impact_scores)) if len(impact_scores) else np.nan,
            **mode_counts,
            **test_eval,

            "channel_noise_name": channel_noise_name,
            "quantum_noise_name": quantum_noise_name,
            "depolarizing_p": getattr(quantum_noise_cfg, "depolarizing_p", 0.0),
        })

        security_rows.append({
            "method": method,
            "security_mode": security_mode,
                    "context_ablation": "+".join(sorted(excluded_context_fields)) if excluded_context_fields else "none",
            "secret_source": secret_manager.label(),
            "attack_mode": attack_mode,
            "round": t,
            "attack_attempts": int(attack_attempts),
            "attack_successes": int(attack_successes),
            "attack_accept_rate": float(attack_successes / max(attack_attempts, 1)) if attack_attempts > 0 else 0.0,
            "context_rejects": int(context_rejects),
            "replay_rejects": int(replay_rejects),
            "aead_or_other_rejects": int(aead_or_other_rejects),
            "accepted_packets": int(accepted_packets),
            "total_packets": int(total_packets),
            "uplink_accept_rate": float(accepted_packets / max(total_packets, 1)),
            "mean_secure_packet_bytes": float(np.mean(secure_packet_bytes)),

            "channel_noise_name": channel_noise_name,
            "quantum_noise_name": quantum_noise_name,
            "depolarizing_p": getattr(quantum_noise_cfg, "depolarizing_p", 0.0),
        })

        print(
            f"[{method} | {security_mode} | {attack_mode} | round {t}] "
            f"acc={test_eval['accuracy']:.4f}, "
            f"attack_success={attack_successes}/{attack_attempts}, "
            f"context={context_rejects}, replay={replay_rejects}, aead={aead_or_other_rejects}"
        )

    return (
        pd.DataFrame(round_rows),
        pd.DataFrame(client_rows),
        pd.DataFrame(security_rows),
        global_theta,
    )
#========================================================






# =============================================================================
# HPC reviewer-revision stages 6-12
# =============================================================================

from dataclasses import dataclass as _stage_dataclass
from collections import defaultdict, deque
import glob


@_stage_dataclass
class CodecAccounting:
    payload_bits: float = 0.0
    index_bits: float = 0.0
    scale_bits: float = 0.0
    metadata_bits: float = 0.0
    security_bits: float = 0.0

    @property
    def total_bits(self):
        return (
            self.payload_bits
            + self.index_bits
            + self.scale_bits
            + self.metadata_bits
            + self.security_bits
        )

    def as_dict(self):
        return {
            "payload_bits": float(self.payload_bits),
            "index_bits": float(self.index_bits),
            "scale_bits": float(self.scale_bits),
            "metadata_bits": float(self.metadata_bits),
            "security_bits": float(self.security_bits),
            "total_bits": float(self.total_bits),
        }


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_auc(y_true, prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return np.nan


def _save_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_stage9_manifest(ctx, out):
    """Archive the exact frozen protocol, dataset files, hashes, and CLI config."""
    manifest = {
        "protocol": FINAL_STAGE9_PROTOCOL,
        "subset_seed": FINAL_SUBSET_SEED,
        "subset_stratification": "target_only" if (ctx.args.num_clients and ctx.args.num_clients > 0) else "target_plus_original_client",
        "synthetic_partition": bool(ctx.args.num_clients and ctx.args.num_clients > 0),
        "num_clients": int(len(ctx.clients)),
        "dirichlet_alpha": float(ctx.args.dirichlet_alpha),
        "train_csv": str(Path(ctx.args.train_csv).resolve()),
        "test_csv": str(Path(ctx.args.test_csv).resolve()),
        "train_csv_sha256": _sha256_file(ctx.args.train_csv),
        "test_csv_sha256": _sha256_file(ctx.args.test_csv),
        "dataset_name": str(ctx.dataset_name),
        "arguments": vars(ctx.args),
        "data_info": ctx.data_info,
        "fixed_method_settings": {
            "learning_rate": float(ctx.spsa_cfg.learning_rate),
            "perturbation": float(ctx.spsa_cfg.perturbation),
            "local_iter": int(ctx.spsa_cfg.local_iter),
            "batch_size": int(ctx.spsa_cfg.batch_size),
            "max_step_norm": float(ctx.spsa_cfg.max_step_norm),
            "warmup_rounds": int(ctx.pred_cfg.warmup_rounds),
            "ema_rho": float(ctx.pred_cfg.ema_rho),
            "residual_small": float(ctx.pred_cfg.residual_small),
            "residual_large": float(ctx.pred_cfg.residual_large),
            "impact_small": float(ctx.pred_cfg.impact_small),
            "impact_large": float(ctx.pred_cfg.impact_large),
            "full_update_period": int(ctx.pred_cfg.full_update_period),
            "prediction_shrink": float(ctx.pred_cfg.prediction_shrink),
            "residual_topk_frac": float(ctx.pred_cfg.residual_topk_frac),
            "residual_quantization_bits": int(ctx.comm_cfg.residual_value_cost),
            "sparse_index_bits": "ceil(log2(parameter_dimension)) per transmitted coordinate",
            "quantization_scale_bits": 32,
        },
    }
    code_path = Path(__file__)
    if code_path.exists():
        manifest["driver_file"] = str(code_path.resolve())
        manifest["driver_sha256"] = _sha256_file(code_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FROZEN_STAGE9_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
    )


def _plot_bar(df, x, y, title, ylabel, out_path, group=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    if group and group in df.columns:
        piv = df.pivot_table(index=x, columns=group, values=y, aggfunc="mean")
        piv.plot(kind="bar", ax=ax)
    else:
        ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Compression codecs for Reviewer Comment 5
# -----------------------------------------------------------------------------

def encode_topk(vec, frac, value_bits=32, metadata_bits=64):
    vec = np.asarray(vec, dtype=float)
    d = int(vec.size)
    k = max(1, int(math.ceil(float(frac) * d)))
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    vals = vec[idx].astype(np.float32)
    recon = np.zeros_like(vec)
    recon[idx] = vals.astype(float)

    idx_bits_each = int(math.ceil(math.log2(max(d, 2))))
    acc = CodecAccounting(
        payload_bits=k * value_bits,
        index_bits=k * idx_bits_each,
        metadata_bits=metadata_bits,
    )
    return recon, idx, vals, acc


def quantize_symmetric(vec, bits, metadata_bits=64):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    vec = np.asarray(vec, dtype=float)
    qmax = (2 ** (bits - 1)) - 1
    vmax = float(np.max(np.abs(vec))) if vec.size else 0.0
    scale = vmax / qmax if vmax > 0 else 1.0
    q = np.clip(np.rint(vec / scale), -qmax, qmax).astype(np.int16)
    recon = q.astype(float) * scale
    acc = CodecAccounting(
        payload_bits=vec.size * bits,
        scale_bits=32,
        metadata_bits=metadata_bits,
    )
    return recon, q, scale, acc


def run_compression_baseline(
    codec_name,
    qnn,
    clients,
    X_test,
    y_test,
    fed_cfg,
    spsa_cfg,
    topk_frac=0.25,
    metadata_bits=64,
):
    """
    Conventional communication baselines using identical local QNN/SPSA training.

    codec_name:
        full_fp32
        topk
        topk_ef
        int8
        int4
    """
    allowed = {"full_fp32", "topk", "topk_ef", "int8", "int4"}
    if codec_name not in allowed:
        raise ValueError(f"codec_name must be one of {sorted(allowed)}")

    rng = np.random.default_rng(fed_cfg.seed)
    n_params = qnn.cfg.n_params
    global_theta = rng.uniform(-0.05, 0.05, size=n_params)

    sizes = np.array([len(yc) for _, _, yc in clients], dtype=float)
    base_weights = sizes / sizes.sum()

    ef_residual = {
        str(cid): np.zeros(n_params, dtype=float)
        for cid, _, _ in clients
    }

    round_rows = []
    client_rows = []

    for t in range(1, fed_cfg.rounds + 1):
        updates = []
        accepted_weights = []
        round_comm = CodecAccounting()

        for j, (cid, Xc, yc) in enumerate(clients):
            local_theta, train_info = spsa_train(
                qnn, Xc, yc, global_theta, spsa_cfg,
                seed=fed_cfg.seed + 1000 * t + j,
            )
            raw_update = wrap_pi(local_theta - global_theta)

            if codec_name == "full_fp32":
                decoded = raw_update.astype(np.float32).astype(float)
                acc = CodecAccounting(
                    payload_bits=n_params * 32,
                    metadata_bits=metadata_bits,
                )

            elif codec_name == "topk":
                decoded, _, _, acc = encode_topk(
                    raw_update, topk_frac, value_bits=32,
                    metadata_bits=metadata_bits,
                )

            elif codec_name == "topk_ef":
                u = wrap_pi(raw_update + ef_residual[str(cid)])
                decoded, _, _, acc = encode_topk(
                    u, topk_frac, value_bits=32,
                    metadata_bits=metadata_bits,
                )
                ef_residual[str(cid)] = wrap_pi(u - decoded)

            elif codec_name == "int8":
                decoded, _, _, acc = quantize_symmetric(
                    raw_update, 8, metadata_bits=metadata_bits
                )

            elif codec_name == "int4":
                decoded, _, _, acc = quantize_symmetric(
                    raw_update, 4, metadata_bits=metadata_bits
                )

            updates.append(decoded)
            accepted_weights.append(base_weights[j])

            round_comm.payload_bits += acc.payload_bits
            round_comm.index_bits += acc.index_bits
            round_comm.scale_bits += acc.scale_bits
            round_comm.metadata_bits += acc.metadata_bits

            client_rows.append({
                "stage": 7 if codec_name == "topk" else 8 if codec_name == "topk_ef" else 9,
                "codec": codec_name,
                "topk_frac": float(topk_frac) if "topk" in codec_name else np.nan,
                "round": t,
                "client_id": str(cid),
                "client_size": int(len(yc)),
                "raw_update_norm": float(np.linalg.norm(raw_update)),
                "decoded_update_norm": float(np.linalg.norm(decoded)),
                "compression_error_norm": float(np.linalg.norm(wrap_pi(raw_update - decoded))),
                **acc.as_dict(),
                **train_info,
            })

        w = np.asarray(accepted_weights, dtype=float)
        w = w / w.sum()
        global_theta = aggregate(global_theta, updates, w)

        metrics = evaluate(qnn, global_theta, X_test, y_test)
        round_rows.append({
            "stage": 7 if codec_name == "topk" else 8 if codec_name == "topk_ef" else 9,
            "codec": codec_name,
            "topk_frac": float(topk_frac) if "topk" in codec_name else np.nan,
            "round": t,
            **metrics,
            **round_comm.as_dict(),
        })

    return pd.DataFrame(round_rows), pd.DataFrame(client_rows), global_theta



# -----------------------------------------------------------------------------
# Literature-inspired communication baselines (mdQFL / ResFed-style)
# -----------------------------------------------------------------------------

def _quantized_topk_residual(vec, frac=0.25, bits=8, metadata_bits=64):
    """Top-k residual with symmetric low-bit value quantization."""
    vec = np.asarray(vec, dtype=float)
    d = int(vec.size)
    k = max(1, int(math.ceil(float(frac) * d)))
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    vals = vec[idx]

    qmax = (2 ** (int(bits) - 1)) - 1
    vmax = float(np.max(np.abs(vals))) if vals.size else 0.0
    scale = vmax / qmax if vmax > 0 else 1.0
    q = np.clip(np.rint(vals / scale), -qmax, qmax).astype(np.int16)
    vals_hat = q.astype(float) * scale

    recon = np.zeros_like(vec)
    recon[idx] = vals_hat
    idx_bits_each = int(math.ceil(math.log2(max(d, 2))))
    acc = CodecAccounting(
        payload_bits=k * int(bits),
        index_bits=k * idx_bits_each,
        scale_bits=32,
        metadata_bits=metadata_bits,
    )
    return recon, idx, q, scale, acc


def run_resfed_style_qfl(
    qnn,
    clients,
    X_test,
    y_test,
    fed_cfg,
    spsa_cfg,
    ema_rho=0.70,
    topk_frac=0.25,
    quant_bits=8,
    metadata_bits=64,
):
    """
    ResFed-style QFL communication baseline.

    This is deliberately labelled *ResFed-style* rather than an exact ResFed
    reproduction.  It uses a synchronized EMA update predictor and, after a
    one-round full-update warm-up, ALWAYS transmits a fixed compressed residual:

        r_i,t = Delta_i,t - Delta_hat_i,t
        r_tilde = Quantize(TopK(r_i,t))
        Delta_tilde_i,t = Delta_hat_i,t + r_tilde

    Unlike predictive_residual_qfl, it has no residual-size thresholds, no
    prediction-only mode, and no adaptive full-update fallback.  This makes it
    a clean residual-compression comparator for the proposed adaptive method.
    """
    if int(quant_bits) not in (4, 8):
        raise ValueError('resfed quant_bits must be 4 or 8')

    rng = np.random.default_rng(fed_cfg.seed)
    n_params = qnn.cfg.n_params
    global_theta = rng.uniform(-0.05, 0.05, size=n_params)
    sizes = np.array([len(yc) for _, _, yc in clients], dtype=float)
    weights = sizes / sizes.sum()
    client_ids = [cid for cid, _, _ in clients]
    predictor = UpdatePredictor(client_ids, n_params, float(ema_rho))

    round_rows, client_rows = [], []
    cumulative_bits = 0.0

    for t in range(1, fed_cfg.rounds + 1):
        updates = []
        round_comm = CodecAccounting()

        for j, (cid, Xc, yc) in enumerate(clients):
            local_theta, train_info = spsa_train(
                qnn, Xc, yc, global_theta, spsa_cfg,
                seed=fed_cfg.seed + 1000 * t + j,
            )
            raw_update = wrap_pi(local_theta - global_theta)

            if t == 1 or not predictor.ready[cid]:
                decoded = raw_update.astype(np.float32).astype(float)
                acc = CodecAccounting(
                    payload_bits=n_params * 32,
                    metadata_bits=metadata_bits,
                )
                mode = 'warmup_full'
                residual_norm = np.nan
            else:
                pred = predictor.predict(cid)
                residual = wrap_pi(raw_update - pred)
                residual_norm = float(np.linalg.norm(residual))
                residual_hat, _, _, _, acc = _quantized_topk_residual(
                    residual,
                    frac=topk_frac,
                    bits=quant_bits,
                    metadata_bits=metadata_bits,
                )
                decoded = wrap_pi(pred + residual_hat)
                mode = 'fixed_residual_topk_quant'

            # Both ends update the synchronized predictor with the reconstructed
            # update, avoiding hidden access to the untransmitted residual.
            predictor.update(cid, decoded)
            updates.append(decoded)

            round_comm.payload_bits += acc.payload_bits
            round_comm.index_bits += acc.index_bits
            round_comm.scale_bits += acc.scale_bits
            round_comm.metadata_bits += acc.metadata_bits

            client_rows.append({
                'method': 'resfed_style_qfl',
                'round': t,
                'client_id': str(cid),
                'client_size': int(len(yc)),
                'mode': mode,
                'topk_frac': float(topk_frac),
                'quant_bits': int(quant_bits),
                'raw_update_norm': float(np.linalg.norm(raw_update)),
                'decoded_update_norm': float(np.linalg.norm(decoded)),
                'residual_norm': residual_norm,
                **acc.as_dict(),
                **train_info,
            })

        global_theta = aggregate(global_theta, updates, weights)
        metrics = evaluate(qnn, global_theta, X_test, y_test)
        cumulative_bits += round_comm.total_bits
        round_rows.append({
            'method': 'resfed_style_qfl',
            'round': t,
            'topk_frac': float(topk_frac),
            'quant_bits': int(quant_bits),
            **metrics,
            **round_comm.as_dict(),
            'cumulative_bits': float(cumulative_bits),
        })
        print(
            f"[resfed_style_qfl | round {t}/{fed_cfg.rounds}] "
            f"acc={metrics['accuracy']:.4f}, auc={metrics.get('auc', np.nan):.4f}, "
            f"bits={round_comm.total_bits:.0f}",
            flush=True,
        )

    return pd.DataFrame(round_rows), pd.DataFrame(client_rows), global_theta


def _mdqfl_cluster_count(n_clients, requested=0):
    """Default mdQFL cluster count: ceil(sqrt(N)), bounded to [1,N]."""
    n_clients = int(n_clients)
    if int(requested) > 0:
        return max(1, min(int(requested), n_clients))
    return max(1, min(int(math.ceil(math.sqrt(n_clients))), n_clients))


def run_mdqfl_baseline(
    qnn,
    clients,
    X_test,
    y_test,
    fed_cfg,
    spsa_cfg,
    n_clusters=0,
    metadata_bits=64,
):
    """
    Concrete mdQFL baseline derived from the uploaded mdQFL paper:
      * K-means grouping from client model parameters,
      * loss-based representative selection within each cluster,
      * one representative performs/transmits the cluster update,
      * cluster-size/data-size weighted qFedAvg aggregation.

    Round 1 follows mdQFL initialization: all clients locally train so that
    client models are available for grouping; those full initial model updates
    are counted as communicated.  Later rounds cluster cached client models and
    communicate only representative updates.  Cluster members adopt the
    representative cluster model (the paper's cluster-model update option).

    This implements one explicit mdQFL configuration rather than every
    personalization/generalization variant in the source paper.
    """
    rng = np.random.default_rng(fed_cfg.seed)
    n_params = qnn.cfg.n_params
    N = len(clients)
    K = _mdqfl_cluster_count(N, n_clusters)
    global_theta = rng.uniform(-0.05, 0.05, size=n_params)

    sizes = {cid: int(len(yc)) for cid, _, yc in clients}
    client_data = {cid: (Xc, yc) for cid, Xc, yc in clients}
    client_order = [cid for cid, _, _ in clients]
    client_states = {cid: global_theta.copy() for cid in client_order}

    round_rows, client_rows = [], []
    cumulative_bits = 0.0
    id_bits = int(math.ceil(math.log2(max(N, 2))))
    cluster_bits = int(math.ceil(math.log2(max(K, 2))))

    for t in range(1, fed_cfg.rounds + 1):
        local_info = {}
        initial_updates = {}

        if t == 1:
            # Initial local training on every client provides the models used
            # for the first grouping step, exactly the costly initialization
            # that mdQFL subsequently amortizes through representative updates.
            for j, cid in enumerate(client_order):
                Xc, yc = client_data[cid]
                theta_i, info = spsa_train(
                    qnn, Xc, yc, global_theta, spsa_cfg,
                    seed=fed_cfg.seed + 1000 * t + j,
                )
                client_states[cid] = theta_i.copy()
                initial_updates[cid] = wrap_pi(theta_i - global_theta)
                local_info[cid] = info

        # Cluster the latest client model states.
        Z = np.vstack([client_states[cid] for cid in client_order])
        if K == N:
            labels = np.arange(N, dtype=int)
        elif K == 1:
            labels = np.zeros(N, dtype=int)
        else:
            km = KMeans(
                n_clusters=K,
                random_state=fed_cfg.seed + t,
                n_init=10,
            )
            labels = km.fit_predict(Z)

        cluster_members = {}
        for j, cid in enumerate(client_order):
            cluster_members.setdefault(int(labels[j]), []).append(cid)

        selected = []
        rep_models = {}
        rep_updates = {}
        rep_train_info = {}

        # Loss-based representative selection, as permitted by mdQFL.
        for cluster_id, members in sorted(cluster_members.items()):
            losses = []
            for cid in members:
                Xc, yc = client_data[cid]
                losses.append((qnn.loss(Xc, yc, client_states[cid]), cid))
            _, rep = min(losses, key=lambda z: z[0])
            selected.append((cluster_id, rep, members))

            if t == 1:
                rep_model = client_states[rep].copy()
                rep_update = initial_updates[rep].copy()
                info = local_info[rep]
            else:
                Xr, yr = client_data[rep]
                rep_model, info = spsa_train(
                    qnn, Xr, yr, global_theta, spsa_cfg,
                    seed=fed_cfg.seed + 1000 * t + client_order.index(rep),
                )
                rep_update = wrap_pi(rep_model - global_theta)

            rep_models[cluster_id] = rep_model
            rep_updates[cluster_id] = rep_update
            rep_train_info[cluster_id] = info

        # Cluster/data-size weighted aggregation of representative updates.
        cluster_sample_sizes = np.array([
            sum(sizes[cid] for cid in members)
            for _, _, members in selected
        ], dtype=float)
        cluster_weights = cluster_sample_sizes / cluster_sample_sizes.sum()
        updates = [rep_updates[cluster_id] for cluster_id, _, _ in selected]
        global_theta = aggregate(global_theta, updates, cluster_weights)

        # mdQFL cluster-model update option: members adopt their representative
        # cluster model for the next grouping round.
        for cluster_id, rep, members in selected:
            for cid in members:
                client_states[cid] = rep_models[cluster_id].copy()

        # Communication accounting (uplink learning communication).
        if t == 1:
            # All N initial client models must be exposed for first grouping.
            payload_bits = N * n_params * 32
            selection_bits = N * id_bits
            active_tx = N
        else:
            active_tx = len(selected)
            payload_bits = active_tx * n_params * 32
            selection_bits = active_tx * (id_bits + cluster_bits)

        # One current-loss scalar per client is conservatively counted for the
        # loss-based selection criterion; metadata is counted per active update.
        loss_metric_bits = N * 32
        md_metadata_bits = active_tx * int(metadata_bits)
        total_bits = float(payload_bits + selection_bits + loss_metric_bits + md_metadata_bits)
        cumulative_bits += total_bits

        metrics = evaluate(qnn, global_theta, X_test, y_test)
        round_rows.append({
            'method': 'mdqfl',
            'round': t,
            'n_clients': N,
            'n_clusters': K,
            'active_clients': int(active_tx),
            'participation_fraction': float(active_tx / max(N, 1)),
            'payload_bits': float(payload_bits),
            'selection_bits': float(selection_bits),
            'loss_metric_bits': float(loss_metric_bits),
            'metadata_bits': float(md_metadata_bits),
            'total_bits': total_bits,
            'cumulative_bits': float(cumulative_bits),
            **metrics,
        })

        selected_ids = {rep for _, rep, _ in selected}
        cluster_lookup = {}
        rep_lookup = {}
        for cluster_id, rep, members in selected:
            for cid in members:
                cluster_lookup[cid] = cluster_id
                rep_lookup[cid] = rep

        for cid in client_order:
            Xc, yc = client_data[cid]
            client_rows.append({
                'method': 'mdqfl',
                'round': t,
                'client_id': str(cid),
                'client_size': sizes[cid],
                'cluster_id': int(cluster_lookup[cid]),
                'representative_id': str(rep_lookup[cid]),
                'selected': int(cid in selected_ids),
                'local_loss_at_cluster_state': float(qnn.loss(Xc, yc, client_states[cid])),
            })

        print(
            f"[mdqfl | round {t}/{fed_cfg.rounds}] N={N}, K={K}, "
            f"active={active_tx}, acc={metrics['accuracy']:.4f}, "
            f"auc={metrics.get('auc', np.nan):.4f}, bits={total_bits:.0f}",
            flush=True,
        )

    return pd.DataFrame(round_rows), pd.DataFrame(client_rows), global_theta

# -----------------------------------------------------------------------------
# Stage 5: controlled S0-S3 security comparison
# -----------------------------------------------------------------------------

def stage5_security_profiles(ctx):
    """
    Compare the four controlled security profiles against the same attack suite.

    S0: ML-KEM + ordinary AEAD
    S1: ML-KEM + context in AEAD AAD
    S2: ML-KEM + context-derived key + AEAD
    S3: full FSB-UP
    """
    out = _ensure_dir(Path(ctx.out_dir) / "stage5_s0_s3_security")

    profiles = [
        "S0_MLKEM_AEAD",
        "S1_MLKEM_CONTEXT_AAD",
        "S2_MLKEM_CONTEXT_KEY",
        "S3_FSB_UP",
    ]
    attacks = [
        "wrong_round",
        "client_swap",
        "stale_model",
        "task_substitution",
        "aggregation_swap",
        "communication_mode_swap",
        "replay",
        "ciphertext_tampering",
    ]

    all_rounds, all_clients, all_sec = [], [], []

    for profile in profiles:
        for attack in attacks:
            print(f"[stage5] profile={profile} | attack={attack}", flush=True)
            r, c, sec, theta = run_qfl_secure_comm(
                method=ctx.method,
                qnn=ctx.qnn,
                clients=ctx.clients,
                X_test=ctx.X_test,
                y_test=ctx.y_test,
                fed_cfg=ctx.fed_cfg,
                spsa_cfg=ctx.spsa_cfg,
                pred_cfg=ctx.pred_cfg,
                comm_cfg=ctx.comm_cfg,
                security_mode=profile,
                secret_source=ctx.secret_source,
                mlkem_algorithm=ctx.mlkem_algorithm,
                attack_mode=attack,
                attack_client_id=ctx.clients[0][0],
                task_name=ctx.dataset_name,
            )

            for df in (r, c, sec):
                df["security_profile"] = profile
                df["attack"] = attack

            run_dir = _ensure_dir(out / profile / attack)
            _save_df(r, run_dir / "round_metrics.csv")
            _save_df(c, run_dir / "client_metrics.csv")
            _save_df(sec, run_dir / "security_metrics.csv")
            np.save(run_dir / "final_theta.npy", theta)

            all_rounds.append(r)
            all_clients.append(c)
            all_sec.append(sec)

    R = pd.concat(all_rounds, ignore_index=True)
    C = pd.concat(all_clients, ignore_index=True)
    S = pd.concat(all_sec, ignore_index=True)

    _save_df(R, out / "all_round_metrics.csv")
    _save_df(C, out / "all_client_metrics.csv")
    _save_df(S, out / "all_security_metrics.csv")

    summary = (
        S.groupby(["security_profile", "attack"], dropna=False)
        .agg(
            attack_attempts=("attack_attempts", "sum"),
            attack_successes=("attack_successes", "sum"),
        )
        .reset_index()
    )
    summary["AAR"] = np.where(
        summary["attack_attempts"] > 0,
        summary["attack_successes"] / summary["attack_attempts"],
        np.nan,
    )
    _save_df(summary, out / "s0_s3_aar_summary.csv")

    # Paper-ready AAR matrix.
    matrix = summary.pivot(
        index="attack",
        columns="security_profile",
        values="AAR"
    )
    matrix.to_csv(out / "s0_s3_aar_matrix.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Attack acceptance rate across S0-S3")
    ax.set_xlabel("Security profile")
    ax.set_ylabel("Attack")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attack acceptance rate (AAR)")
    fig.tight_layout()
    fig.savefig(out / "s0_s3_aar_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


# -----------------------------------------------------------------------------
# Stage 6: semantic-field ablation
# -----------------------------------------------------------------------------

def stage6_security_ablation(ctx):
    out = _ensure_dir(Path(ctx.out_dir) / "stage6_security_ablation")
    qnn, clients, X_test, y_test = ctx.qnn, ctx.clients, ctx.X_test, ctx.y_test

    ablations = [
        ("round_id", "wrong_round"),
        ("client_id", "client_swap"),
        ("model_hash", "stale_model"),
        ("task", "task_substitution"),
        ("aggregation", "aggregation_swap"),
        ("comm_method", "communication_mode_swap"),
    ]

    all_rounds, all_clients, all_sec = [], [], []

    for removed_field, attack in ablations:
        for label, excluded in [
            ("intact_context", frozenset()),
            (f"without_{removed_field}", frozenset({removed_field})),
        ]:
            print(f"[stage6] {label} | attack={attack}", flush=True)
            r, c, s, theta = run_qfl_secure_comm(
                method=ctx.method,
                qnn=qnn,
                clients=clients,
                X_test=X_test,
                y_test=y_test,
                fed_cfg=ctx.fed_cfg,
                spsa_cfg=ctx.spsa_cfg,
                pred_cfg=ctx.pred_cfg,
                comm_cfg=ctx.comm_cfg,
                security_mode="S3_FSB_UP",
                secret_source=ctx.secret_source,
                mlkem_algorithm=ctx.mlkem_algorithm,
                attack_mode=attack,
                attack_client_id=clients[0][0],
                task_name=ctx.dataset_name,
                excluded_context_fields=excluded,
            )
            for df in (r, c, s):
                df["ablation_name"] = label
                df["removed_field"] = removed_field
                df["matched_attack"] = attack

            run_dir = _ensure_dir(out / removed_field / label)
            _save_df(r, run_dir / "round_metrics.csv")
            _save_df(c, run_dir / "client_metrics.csv")
            _save_df(s, run_dir / "security_metrics.csv")
            np.save(run_dir / "final_theta.npy", theta)
            all_rounds.append(r)
            all_clients.append(c)
            all_sec.append(s)

    R = pd.concat(all_rounds, ignore_index=True)
    C = pd.concat(all_clients, ignore_index=True)
    S = pd.concat(all_sec, ignore_index=True)
    _save_df(R, out / "all_round_metrics.csv")
    _save_df(C, out / "all_client_metrics.csv")
    _save_df(S, out / "all_security_metrics.csv")

    # Robust AAR summary: use per-run aggregate columns when available.
    group_cols = ["removed_field", "matched_attack", "ablation_name"]
    agg = {}
    if "attack_attempts" in S.columns:
        agg["attack_attempts"] = "sum"
    if "attack_successes" in S.columns:
        agg["attack_successes"] = "sum"
    if "accepted_packets" in S.columns:
        agg["accepted_packets"] = "sum"
    if "total_packets" in S.columns:
        agg["total_packets"] = "sum"

    if agg:
        summary = S.groupby(group_cols, dropna=False).agg(agg).reset_index()
        if {"attack_attempts", "attack_successes"}.issubset(summary.columns):
            summary["AAR"] = np.where(
                summary["attack_attempts"] > 0,
                summary["attack_successes"] / summary["attack_attempts"],
                np.nan,
            )
    else:
        # Fall back to row-level fields if notebook version differs.
        summary = S.groupby(group_cols, dropna=False).size().reset_index(name="rows")
        summary["AAR"] = np.nan

    _save_df(summary, out / "ablation_aar_summary.csv")
    if "AAR" in summary.columns and summary["AAR"].notna().any():
        _plot_bar(
            summary,
            "removed_field",
            "AAR",
            "FSB-UP context-field ablation",
            "Attack acceptance rate (AAR)",
            out / "ablation_aar.png",
            group="ablation_name",
        )

    return summary


# -----------------------------------------------------------------------------
# Stages 7-9: compression baselines
# -----------------------------------------------------------------------------

def _run_codec_sweep(ctx, codecs, stage, topk_fracs=(0.10, 0.25, 0.50, 0.75)):
    out = _ensure_dir(Path(ctx.out_dir) / f"stage{stage}_compression")
    all_r, all_c = [], []

    for codec in codecs:
        fracs = topk_fracs if "topk" in codec else (np.nan,)
        for frac in fracs:
            tag = f"{codec}_k{frac:.2f}" if "topk" in codec else codec
            print(f"[stage{stage}] {tag}", flush=True)
            r, c, theta = run_compression_baseline(
                codec_name=codec,
                qnn=ctx.qnn,
                clients=ctx.clients,
                X_test=ctx.X_test,
                y_test=ctx.y_test,
                fed_cfg=ctx.fed_cfg,
                spsa_cfg=ctx.spsa_cfg,
                topk_frac=0.25 if np.isnan(frac) else float(frac),
                metadata_bits=int(ctx.comm_cfg.metadata_cost),
            )
            r["run_tag"] = tag
            c["run_tag"] = tag
            d = _ensure_dir(out / tag)
            _save_df(r, d / "round_metrics.csv")
            _save_df(c, d / "client_metrics.csv")
            np.save(d / "final_theta.npy", theta)
            all_r.append(r)
            all_c.append(c)

    R = pd.concat(all_r, ignore_index=True)
    C = pd.concat(all_c, ignore_index=True)
    _save_df(R, out / "all_round_metrics.csv")
    _save_df(C, out / "all_client_metrics.csv")
    return R, C


def stage7_topk(ctx):
    return _run_codec_sweep(ctx, ["topk"], 7)


def stage8_topk_error_feedback(ctx):
    return _run_codec_sweep(ctx, ["topk_ef"], 8)


def stage9_literature_baselines_only(ctx):
    """
    Run only mdQFL-style and ResFed-style QFL using the same runtime context
    as the current Stage-9 experiment. Results are added to the existing
    stage9_compression directory and do not overwrite the existing aggregate
    Stage-9 files for FP32/INT8/INT4/proposed methods.
    """
    out = _ensure_dir(Path(ctx.out_dir) / "stage9_compression")
    _write_stage9_manifest(ctx, out)
    baseline_rounds, baseline_clients = [], []

    print("[stage9-only] communication baseline: mdqfl", flush=True)
    mr, mc, mtheta = run_mdqfl_baseline(
        qnn=ctx.qnn, clients=ctx.clients, X_test=ctx.X_test, y_test=ctx.y_test,
        fed_cfg=ctx.fed_cfg, spsa_cfg=ctx.spsa_cfg,
        n_clusters=ctx.args.mdqfl_clusters,
        metadata_bits=int(ctx.comm_cfg.metadata_cost),
    )
    mr["run_tag"] = "mdqfl"
    mc["run_tag"] = "mdqfl"
    md = _ensure_dir(out / "reference_mdqfl")
    _save_df(mr, md / "round_metrics.csv")
    _save_df(mc, md / "client_metrics.csv")
    np.save(md / "final_theta.npy", mtheta)
    baseline_rounds.append(mr); baseline_clients.append(mc)

    print("[stage9-only] communication baseline: resfed_style_qfl", flush=True)
    rr, rc, rtheta = run_resfed_style_qfl(
        qnn=ctx.qnn, clients=ctx.clients, X_test=ctx.X_test, y_test=ctx.y_test,
        fed_cfg=ctx.fed_cfg, spsa_cfg=ctx.spsa_cfg,
        ema_rho=ctx.args.ema_rho,
        topk_frac=ctx.args.resfed_topk_frac,
        quant_bits=ctx.args.resfed_bits,
        metadata_bits=int(ctx.comm_cfg.metadata_cost),
    )
    rr["run_tag"] = "resfed_style_qfl"
    rc["run_tag"] = "resfed_style_qfl"
    rd = _ensure_dir(out / "reference_resfed_style_qfl")
    _save_df(rr, rd / "round_metrics.csv")
    _save_df(rc, rd / "client_metrics.csv")
    np.save(rd / "final_theta.npy", rtheta)
    baseline_rounds.append(rr); baseline_clients.append(rc)

    _save_df(pd.concat(baseline_rounds, ignore_index=True),
             out / "literature_baseline_round_metrics.csv")
    _save_df(pd.concat(baseline_clients, ignore_index=True),
             out / "literature_baseline_client_metrics.csv")
    print(f"[stage9-only] completed mdQFL + ResFed-style -> {out}", flush=True)
    return pd.concat(baseline_rounds, ignore_index=True), pd.concat(baseline_clients, ignore_index=True)


def stage9_quantization_and_reference(ctx):
    out = _ensure_dir(Path(ctx.out_dir) / "stage9_compression")
    _write_stage9_manifest(ctx, out)
    R, C = _run_codec_sweep(ctx, ["full_fp32", "int8", "int4"], 9)

    if getattr(ctx.args, "include_literature_baselines", False):
        stage9_literature_baselines_only(ctx)

    # Reference proposed communication methods under clean/no-attack S3.
    ref_rounds = []
    ref_clients = []
    reference_methods = [
        "predictive_residual_qfl",
        "aggregation_impact_qfl",
    ]
    if getattr(ctx.args, "include_telesync", False):
        reference_methods.append("telesync_style_qfl")

    for method in reference_methods:
        print(f"[stage9] proposed reference: {method}", flush=True)
        rr, cc, ss, theta = run_qfl_secure_comm(
            method=method,
            qnn=ctx.qnn,
            clients=ctx.clients,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            fed_cfg=ctx.fed_cfg,
            spsa_cfg=ctx.spsa_cfg,
            pred_cfg=ctx.pred_cfg,
            comm_cfg=ctx.comm_cfg,
            security_mode="S3_FSB_UP",
            secret_source=ctx.secret_source,
            mlkem_algorithm=ctx.mlkem_algorithm,
            attack_mode="none",
            task_name=ctx.dataset_name,
        )
        rr["run_tag"] = method
        cc["run_tag"] = method
        d = _ensure_dir(out / f"reference_{method}")
        _save_df(rr, d / "round_metrics.csv")
        _save_df(cc, d / "client_metrics.csv")
        _save_df(ss, d / "security_metrics.csv")
        np.save(d / "final_theta.npy", theta)
        ref_rounds.append(rr)
        ref_clients.append(cc)

    if ref_rounds:
        _save_df(pd.concat(ref_rounds, ignore_index=True), out / "reference_all_round_metrics.csv")
        _save_df(pd.concat(ref_clients, ignore_index=True), out / "reference_all_client_metrics.csv")

    return R, C


# -----------------------------------------------------------------------------
# Trace-driven V2X support for Reviewer Comment 4
# -----------------------------------------------------------------------------

def _boolify(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(int(v))
    return str(v).strip().lower() in {"1", "true", "yes", "y", "lost", "down", "disconnected"}


def load_v2x_trace_schedule(
    trace_csv,
    client_col="client_id",
    time_col="timestamp",
    rounds=20,
    round_duration_ms=1000.0,
):
    """
    Build a deterministic client/round network schedule.

    Preferred trace columns:
      client_id                 required
      timestamp                 preferred
      delay_ms                  optional
      packet_lost               optional
      connected                 optional
      serving_rsu               optional

    If packet_lost is absent, loss is inferred only when connected==False.
    If delay_ms is absent, delay_rounds defaults to zero.
    Handover is inferred from serving_rsu changes within a client trajectory.

    This function intentionally does NOT interpret encoded ML features x1..x4
    as physical mobility variables.
    """
    trace = pd.read_csv(trace_csv)
    if client_col not in trace.columns:
        raise ValueError(f"Trace file must contain '{client_col}'")

    if time_col in trace.columns:
        trace = trace.sort_values([client_col, time_col]).copy()
    else:
        trace = trace.copy()
        trace["_trace_order"] = trace.groupby(client_col).cumcount()
        time_col = "_trace_order"

    if "serving_rsu" in trace.columns:
        trace["handover"] = (
            trace.groupby(client_col)["serving_rsu"]
            .transform(lambda s: s.astype(str).ne(s.astype(str).shift(1)))
            .fillna(False)
        )
        # First observation is not a handover.
        first_idx = trace.groupby(client_col).head(1).index
        trace.loc[first_idx, "handover"] = False
    else:
        trace["handover"] = False

    # Map each client's ordered observations uniformly onto FL rounds.
    schedule = {}
    for cid, g in trace.groupby(client_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        idxs = np.linspace(0, n - 1, rounds).round().astype(int)
        for r, idx in enumerate(idxs, start=1):
            row = g.iloc[int(idx)]
            connected = True
            if "connected" in g.columns:
                connected = _boolify(row["connected"])
            lost = False
            if "packet_lost" in g.columns:
                lost = _boolify(row["packet_lost"])
            elif not connected:
                lost = True

            delay_ms = float(row["delay_ms"]) if "delay_ms" in g.columns and pd.notna(row["delay_ms"]) else 0.0
            delay_rounds = int(max(0, math.ceil(delay_ms / max(round_duration_ms, 1e-9))))
            handover = bool(row["handover"])

            schedule[(str(cid), r)] = {
                "connected": connected,
                "packet_lost": lost,
                "delay_ms": delay_ms,
                "delay_rounds": delay_rounds,
                "handover": handover,
                "serving_rsu": str(row["serving_rsu"]) if "serving_rsu" in g.columns else "",
            }

    return schedule, trace


def _method_update_decision(
    method,
    cid,
    local_idx,
    t,
    actual_update,
    weights,
    predictor,
    client_update_memory,
    memory_ready,
    pred_cfg,
    comm_cfg,
):
    n_params = len(actual_update)

    if method == "tuned_qfl":
        return actual_update.copy(), "full", n_params, n_params * comm_cfg.full_value_cost

    if method == "predictive_residual_qfl":
        pred = predictor.predict(cid)
        residual = wrap_pi(actual_update - pred)
        rn = float(np.linalg.norm(residual))
        force_full = (
            t <= pred_cfg.warmup_rounds
            or (pred_cfg.full_update_period > 0 and t % pred_cfg.full_update_period == 0)
        )
        if force_full:
            recon, mode, sent = actual_update.copy(), "full", n_params
        elif rn <= pred_cfg.residual_small:
            recon, mode, sent = pred_cfg.prediction_shrink * pred, "prediction_only", 0
        elif rn <= pred_cfg.residual_large:
            corr, sent = sparsify_topk(residual, pred_cfg.residual_topk_frac)
            recon, mode = wrap_pi(pred + corr), "residual"
        else:
            recon, mode, sent = actual_update.copy(), "full_fallback", n_params
        predictor.update(cid, recon)
        bits = sent * (comm_cfg.residual_value_cost if mode == "residual" else comm_cfg.full_value_cost) + comm_cfg.metadata_cost
        if mode == "prediction_only":
            bits = comm_cfg.metadata_cost
        return recon, mode, sent, bits

    if method == "aggregation_impact_qfl":
        prev = client_update_memory[cid]
        if memory_ready[cid]:
            impact = wrap_pi(actual_update - prev)
            score = float(weights[local_idx] * np.linalg.norm(impact))
        else:
            impact = actual_update.copy()
            score = np.inf

        force_full = (
            not memory_ready[cid]
            or t <= pred_cfg.warmup_rounds
            or (pred_cfg.full_update_period > 0 and t % pred_cfg.full_update_period == 0)
        )
        if force_full:
            recon, mode, sent = actual_update.copy(), "full", n_params
        elif score <= pred_cfg.impact_small:
            recon, mode, sent = pred_cfg.memory_shrink * prev, "memory_reuse", 0
        elif score <= pred_cfg.impact_large:
            corr, sent = sparsify_topk(impact, pred_cfg.residual_topk_frac)
            recon, mode = wrap_pi(prev + corr), "impact_residual"
        else:
            recon, mode, sent = actual_update.copy(), "full_fallback", n_params

        client_update_memory[cid] = recon.copy()
        memory_ready[cid] = True
        if sent == 0:
            bits = comm_cfg.metadata_cost
        elif mode == "impact_residual":
            bits = sent * comm_cfg.residual_value_cost + comm_cfg.metadata_cost
        else:
            bits = sent * comm_cfg.full_value_cost + comm_cfg.metadata_cost
        return recon, mode, sent, bits

    if method == "telesync_style_qfl":
        recon, tele = telesync_transport_update(
            actual_update,
            seed=12345 + 1000 * t + local_idx,
        )
        bits = (
            tele["telesync_correction_bits"]
            + tele["telesync_side_bits"]
            + comm_cfg.metadata_cost
        )
        return recon, "telesync_state_sync", 0, bits

    raise ValueError(f"Unknown method: {method}")


def run_trace_driven_qfl(
    method,
    qnn,
    clients,
    X_test,
    y_test,
    fed_cfg,
    spsa_cfg,
    pred_cfg,
    comm_cfg,
    trace_schedule,
):
    """
    FL simulation with deterministic trace-driven loss/delay/handover.

    Updates that are delayed are queued and delivered at round t + delay_rounds.
    Lost/disconnected updates are not aggregated.
    Staleness is recorded as current_round - creation_round.
    """
    rng = np.random.default_rng(fed_cfg.seed)
    client_ids = [str(cid) for cid, _, _ in clients]
    n_params = qnn.cfg.n_params
    theta = rng.uniform(-0.05, 0.05, size=n_params)

    sizes = np.array([len(yc) for _, _, yc in clients], dtype=float)
    weights = sizes / sizes.sum()

    predictor = UpdatePredictor(client_ids, n_params, pred_cfg.ema_rho)
    memory = {cid: np.zeros(n_params, dtype=float) for cid in client_ids}
    ready = {cid: False for cid in client_ids}

    pending = defaultdict(list)
    round_rows, client_rows = [], []

    for t in range(1, fed_cfg.rounds + 1):
        delivered_updates = []
        delivered_weights = []
        delivered_staleness = []

        # Deliver queued packets first.
        for item in pending.pop(t, []):
            delivered_updates.append(item["update"])
            delivered_weights.append(item["weight"])
            delivered_staleness.append(t - item["created_round"])

        generated_bits = 0.0
        dropped = 0
        delayed = 0
        handovers = 0

        for j, (cid_raw, Xc, yc) in enumerate(clients):
            cid = str(cid_raw)
            local_theta, train_info = spsa_train(
                qnn, Xc, yc, theta, spsa_cfg,
                seed=fed_cfg.seed + 1000 * t + j,
            )
            actual = wrap_pi(local_theta - theta)
            recon, mode, sent, bits = _method_update_decision(
                method, cid, j, t, actual, weights, predictor,
                memory, ready, pred_cfg, comm_cfg,
            )
            generated_bits += float(bits)

            ev = trace_schedule.get((cid, t), {
                "connected": True,
                "packet_lost": False,
                "delay_ms": 0.0,
                "delay_rounds": 0,
                "handover": False,
                "serving_rsu": "",
            })
            lost = bool(ev["packet_lost"]) or not bool(ev["connected"])
            dr = int(ev["delay_rounds"])
            if ev["handover"]:
                handovers += 1

            status = "delivered"
            if lost:
                dropped += 1
                status = "lost_or_disconnected"
            elif dr > 0:
                delayed += 1
                status = "queued"
                delivery_round = t + dr
                if delivery_round <= fed_cfg.rounds:
                    pending[delivery_round].append({
                        "update": recon.copy(),
                        "weight": float(weights[j]),
                        "created_round": t,
                    })
            else:
                delivered_updates.append(recon)
                delivered_weights.append(float(weights[j]))
                delivered_staleness.append(0)

            client_rows.append({
                "method": method,
                "round": t,
                "client_id": cid,
                "mode": mode,
                "network_status": status,
                "connected": int(bool(ev["connected"])),
                "packet_lost": int(bool(ev["packet_lost"])),
                "delay_ms": float(ev["delay_ms"]),
                "delay_rounds": int(dr),
                "handover": int(bool(ev["handover"])),
                "serving_rsu": ev["serving_rsu"],
                "generated_bits": float(bits),
                "sent_values": int(sent),
                **train_info,
            })

        if delivered_updates:
            w = np.asarray(delivered_weights, dtype=float)
            w /= w.sum()
            theta = aggregate(theta, delivered_updates, w)

        m = evaluate(qnn, theta, X_test, y_test)
        round_rows.append({
            "method": method,
            "round": t,
            **m,
            "generated_bits": generated_bits,
            "delivered_updates": len(delivered_updates),
            "dropped_updates": dropped,
            "delayed_updates": delayed,
            "handovers": handovers,
            "mean_staleness": float(np.mean(delivered_staleness)) if delivered_staleness else np.nan,
            "max_staleness": float(np.max(delivered_staleness)) if delivered_staleness else np.nan,
            "pending_queue": int(sum(len(v) for v in pending.values())),
        })

    return pd.DataFrame(round_rows), pd.DataFrame(client_rows), theta


def stage10_trace_v2x(ctx):
    if not ctx.trace_csv:
        raise ValueError("Stage 10 requires --trace-csv")
    out = _ensure_dir(Path(ctx.out_dir) / "stage10_trace_v2x")
    schedule, trace = load_v2x_trace_schedule(
        ctx.trace_csv,
        client_col=ctx.trace_client_col,
        time_col=ctx.trace_time_col,
        rounds=ctx.fed_cfg.rounds,
        round_duration_ms=ctx.round_duration_ms,
    )
    _save_df(trace, out / "trace_used.csv")

    all_r, all_c = [], []
    for method in ctx.trace_methods:
        print(f"[stage10] trace-driven network: {method}", flush=True)
        r, c, theta = run_trace_driven_qfl(
            method, ctx.qnn, ctx.clients, ctx.X_test, ctx.y_test,
            ctx.fed_cfg, ctx.spsa_cfg, ctx.pred_cfg, ctx.comm_cfg, schedule,
        )
        r["experiment"] = "trace_delay_loss_disconnection"
        c["experiment"] = "trace_delay_loss_disconnection"
        d = _ensure_dir(out / method)
        _save_df(r, d / "round_metrics.csv")
        _save_df(c, d / "client_metrics.csv")
        np.save(d / "final_theta.npy", theta)
        all_r.append(r); all_c.append(c)

    _save_df(pd.concat(all_r, ignore_index=True), out / "all_round_metrics.csv")
    _save_df(pd.concat(all_c, ignore_index=True), out / "all_client_metrics.csv")


def stage11_handover_staleness(ctx):
    if not ctx.trace_csv:
        raise ValueError("Stage 11 requires --trace-csv")
    out = _ensure_dir(Path(ctx.out_dir) / "stage11_handover_staleness")
    schedule, trace = load_v2x_trace_schedule(
        ctx.trace_csv,
        client_col=ctx.trace_client_col,
        time_col=ctx.trace_time_col,
        rounds=ctx.fed_cfg.rounds,
        round_duration_ms=ctx.round_duration_ms,
    )

    all_r, all_c = [], []
    for method in ctx.trace_methods:
        print(f"[stage11] handover/staleness: {method}", flush=True)
        r, c, theta = run_trace_driven_qfl(
            method, ctx.qnn, ctx.clients, ctx.X_test, ctx.y_test,
            ctx.fed_cfg, ctx.spsa_cfg, ctx.pred_cfg, ctx.comm_cfg, schedule,
        )
        r["experiment"] = "handover_staleness"
        c["experiment"] = "handover_staleness"
        d = _ensure_dir(out / method)
        _save_df(r, d / "round_metrics.csv")
        _save_df(c, d / "client_metrics.csv")
        np.save(d / "final_theta.npy", theta)
        all_r.append(r); all_c.append(c)

    R = pd.concat(all_r, ignore_index=True)
    C = pd.concat(all_c, ignore_index=True)
    _save_df(R, out / "all_round_metrics.csv")
    _save_df(C, out / "all_client_metrics.csv")

    # Staleness summary
    cols = [c for c in ["accuracy", "balanced_accuracy", "f1", "auc", "mean_staleness", "max_staleness"] if c in R.columns]
    if cols:
        summary = R.groupby("method")[cols].agg(["mean", "std"]).reset_index()
        summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in summary.columns]
        _save_df(summary, out / "handover_staleness_summary.csv")


# -----------------------------------------------------------------------------
# Stage 12: consolidation
# -----------------------------------------------------------------------------

def stage12_consolidate(ctx):
    root = Path(ctx.out_dir)
    out = _ensure_dir(root / "stage12_consolidated")

    # Compression runs
    comp_files = list(root.glob("stage7_compression/**/round_metrics.csv")) \
               + list(root.glob("stage8_compression/**/round_metrics.csv")) \
               + list(root.glob("stage9_compression/**/round_metrics.csv"))

    comp = []
    for f in comp_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            comp.append(df)
        except Exception as e:
            print("[stage12] skip", f, e)

    if comp:
        C = pd.concat(comp, ignore_index=True)
        _save_df(C, out / "all_compression_round_metrics.csv")

        metric_cols = [x for x in ["accuracy", "balanced_accuracy", "f1", "auc"] if x in C.columns]
        if "total_bits" in C.columns:
            C["cumulative_bits"] = C.groupby("run_tag" if "run_tag" in C.columns else "codec")["total_bits"].cumsum()

        key = "run_tag" if "run_tag" in C.columns else "codec"
        final = C.sort_values("round").groupby(key, as_index=False).tail(1).copy()
        keep = [key] + metric_cols + [x for x in ["total_bits", "cumulative_bits"] if x in final.columns]
        _save_df(final[keep], out / "communication_final_summary.csv")

        # Matched-budget pairs: each proposed reference vs nearest conventional baseline.
        if "cumulative_bits" in final.columns:
            proposed_mask = final[key].astype(str).str.contains(
                "predictive_residual|aggregation_impact|teleport_impact", regex=True
            )
            proposed = final[proposed_mask].copy()
            baseline = final[~proposed_mask].copy()
            rows = []
            for _, p in proposed.iterrows():
                if len(baseline) == 0:
                    continue
                idx = (baseline["cumulative_bits"] - p["cumulative_bits"]).abs().idxmin()
                b = baseline.loc[idx]
                row = {
                    "proposed_method": p[key],
                    "proposed_bits": p["cumulative_bits"],
                    "matched_baseline": b[key],
                    "baseline_bits": b["cumulative_bits"],
                    "budget_gap_bits": abs(b["cumulative_bits"] - p["cumulative_bits"]),
                }
                for m in metric_cols:
                    row[f"proposed_{m}"] = p.get(m, np.nan)
                    row[f"baseline_{m}"] = b.get(m, np.nan)
                    row[f"delta_{m}"] = p.get(m, np.nan) - b.get(m, np.nan)
                rows.append(row)
            if rows:
                _save_df(pd.DataFrame(rows), out / "matched_budget_summary.csv")

    # S0-S3 controlled security comparison
    sec_profiles = root / "stage5_s0_s3_security" / "s0_s3_aar_summary.csv"
    if sec_profiles.exists():
        pd.read_csv(sec_profiles).to_csv(
            out / "s0_s3_security_summary.csv", index=False
        )

    # Security ablation
    sec = root / "stage6_security_ablation" / "ablation_aar_summary.csv"
    if sec.exists():
        pd.read_csv(sec).to_csv(out / "security_ablation_summary.csv", index=False)

    # V2X
    v2x_files = [
        root / "stage10_trace_v2x" / "all_round_metrics.csv",
        root / "stage11_handover_staleness" / "all_round_metrics.csv",
    ]
    vdfs = []
    for f in v2x_files:
        if f.exists():
            d = pd.read_csv(f)
            d["source"] = f.parent.name
            vdfs.append(d)
    if vdfs:
        V = pd.concat(vdfs, ignore_index=True)
        _save_df(V, out / "all_v2x_round_metrics.csv")
        metrics = [c for c in [
            "accuracy", "balanced_accuracy", "f1", "auc",
            "generated_bits", "delivered_updates", "dropped_updates",
            "delayed_updates", "handovers", "mean_staleness", "max_staleness"
        ] if c in V.columns]
        if metrics:
            summary = V.groupby(["source", "method"])[metrics].agg(["mean", "std"]).reset_index()
            summary.columns = [
                "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]
            _save_df(summary, out / "v2x_statistical_summary.csv")

    print(f"[stage12] consolidated outputs: {out}", flush=True)



# -----------------------------------------------------------------------------
# Optional synthetic client scaling / non-IID repartitioning
# -----------------------------------------------------------------------------

def make_dirichlet_clients(X, y, n_clients, alpha, seed, min_client_size=20):
    """
    Repartition the selected training subset into n_clients synthetic FL clients.

    alpha controls label heterogeneity:
      large alpha -> closer to IID
      small alpha -> more non-IID
    Recommended reviewer scalability setting: alpha=0.5.

    The repartitioning is deterministic for a fixed seed.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_clients = int(n_clients)
    alpha = float(alpha)
    if n_clients <= 0:
        raise ValueError("n_clients must be > 0")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    # Retry to avoid pathologically tiny clients.
    for attempt in range(200):
        buckets = [[] for _ in range(n_clients)]

        for cls in classes:
            idx = np.where(y == cls)[0].copy()
            rng.shuffle(idx)

            proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
            counts = rng.multinomial(len(idx), proportions)

            start = 0
            for cid, count in enumerate(counts):
                if count > 0:
                    buckets[cid].extend(idx[start:start + count].tolist())
                start += count

        sizes = np.array([len(b) for b in buckets])
        if np.all(sizes >= min_client_size):
            break
    else:
        raise RuntimeError(
            f"Could not create {n_clients} clients with min size "
            f"{min_client_size} after 200 attempts. Increase train subset size "
            f"or reduce the number of clients."
        )

    clients = []
    for cid, idxs in enumerate(buckets):
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)
        clients.append((cid, X[idxs], y[idxs]))

    return clients


def client_partition_summary(clients):
    rows = []
    for cid, Xc, yc in clients:
        values, counts = np.unique(yc, return_counts=True)
        row = {
            "client_id": cid,
            "n_samples": int(len(yc)),
            "positive_rate": float(np.mean(yc)) if len(yc) else np.nan,
        }
        for v, c in zip(values, counts):
            row[f"class_{v}_count"] = int(c)
        rows.append(row)
    return pd.DataFrame(rows)


def stage4_scalability(ctx):
    """
    Lightweight scalability experiment.
    Runs representative tuned_qfl and aggregation_impact_qfl methods for the
    current client partition and records utility, communication, and runtime.
    Invoke repeatedly with --num-clients 5/10/20/50 and --dirichlet-alpha 0.5.
    """
    out = _ensure_dir(Path(ctx.out_dir) / "stage4_scalability")
    methods = ["tuned_qfl", "aggregation_impact_qfl"]
    rows = []

    part = client_partition_summary(ctx.clients)
    _save_df(part, out / f"client_partition_N{len(ctx.clients)}_alpha{ctx.args.dirichlet_alpha}.csv")

    for method in methods:
        print(
            f"[stage4] clients={len(ctx.clients)} alpha={ctx.args.dirichlet_alpha} "
            f"method={method}",
            flush=True,
        )
        t0 = time.perf_counter()
        r, c, sec, theta = run_qfl_secure_comm(
            method=method,
            qnn=ctx.qnn,
            clients=ctx.clients,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            fed_cfg=ctx.fed_cfg,
            spsa_cfg=ctx.spsa_cfg,
            pred_cfg=ctx.pred_cfg,
            comm_cfg=ctx.comm_cfg,
            security_mode="S3_FSB_UP",
            secret_source=ctx.secret_source,
            mlkem_algorithm=ctx.mlkem_algorithm,
            attack_mode="none",
            task_name=ctx.dataset_name,
        )
        runtime = time.perf_counter() - t0

        tag = f"N{len(ctx.clients)}_alpha{ctx.args.dirichlet_alpha}_{method}"
        d = _ensure_dir(out / tag)
        _save_df(r, d / "round_metrics.csv")
        _save_df(c, d / "client_metrics.csv")
        _save_df(sec, d / "security_metrics.csv")
        np.save(d / "final_theta.npy", theta)

        last = r.sort_values("round").iloc[-1]
        total_bits = float(r["bits"].sum()) if "bits" in r.columns else (
            float(r["round_bits"].sum()) if "round_bits" in r.columns else np.nan
        )
        rows.append({
            "seed": ctx.args.seed,
            "n_clients": len(ctx.clients),
            "dirichlet_alpha": ctx.args.dirichlet_alpha,
            "method": method,
            "rounds": ctx.fed_cfg.rounds,
            "runtime_seconds": runtime,
            "seconds_per_round": runtime / max(ctx.fed_cfg.rounds, 1),
            "total_bits": total_bits,
            "bits_per_client_per_round": (
                total_bits / (len(ctx.clients) * ctx.fed_cfg.rounds)
                if np.isfinite(total_bits) else np.nan
            ),
            "accuracy": last.get("accuracy", np.nan),
            "balanced_accuracy": last.get("balanced_accuracy", np.nan),
            "f1": last.get("f1", np.nan),
            "auc": last.get("auc", np.nan),
        })

    # Optional literature-inspired scalability baselines. These are disabled by
    # default so existing Stage-4 behavior is unchanged.
    if getattr(ctx.args, "include_literature_baselines", False):
        literature = [
            ("mdqfl", lambda: run_mdqfl_baseline(
                qnn=ctx.qnn, clients=ctx.clients, X_test=ctx.X_test, y_test=ctx.y_test,
                fed_cfg=ctx.fed_cfg, spsa_cfg=ctx.spsa_cfg,
                n_clusters=ctx.args.mdqfl_clusters,
                metadata_bits=int(ctx.comm_cfg.metadata_cost))),
            ("resfed_style_qfl", lambda: run_resfed_style_qfl(
                qnn=ctx.qnn, clients=ctx.clients, X_test=ctx.X_test, y_test=ctx.y_test,
                fed_cfg=ctx.fed_cfg, spsa_cfg=ctx.spsa_cfg,
                ema_rho=ctx.args.ema_rho, topk_frac=ctx.args.resfed_topk_frac,
                quant_bits=ctx.args.resfed_bits,
                metadata_bits=int(ctx.comm_cfg.metadata_cost))),
        ]
        for method, runner in literature:
            print(f"[stage4] clients={len(ctx.clients)} alpha={ctx.args.dirichlet_alpha} method={method}", flush=True)
            t0 = time.perf_counter()
            r, c, theta = runner()
            runtime = time.perf_counter() - t0
            tag = f"N{len(ctx.clients)}_alpha{ctx.args.dirichlet_alpha}_{method}"
            d = _ensure_dir(out / tag)
            _save_df(r, d / "round_metrics.csv")
            _save_df(c, d / "client_metrics.csv")
            np.save(d / "final_theta.npy", theta)
            last = r.sort_values("round").iloc[-1]
            total_bits = float(last.get("cumulative_bits", np.nan))
            rows.append({
                "seed": ctx.args.seed,
                "n_clients": len(ctx.clients),
                "dirichlet_alpha": ctx.args.dirichlet_alpha,
                "method": method,
                "rounds": ctx.fed_cfg.rounds,
                "runtime_seconds": runtime,
                "seconds_per_round": runtime / max(ctx.fed_cfg.rounds, 1),
                "total_bits": total_bits,
                "bits_per_client_per_round": (
                    total_bits / (len(ctx.clients) * ctx.fed_cfg.rounds)
                    if np.isfinite(total_bits) else np.nan
                ),
                "accuracy": last.get("accuracy", np.nan),
                "balanced_accuracy": last.get("balanced_accuracy", np.nan),
                "f1": last.get("f1", np.nan),
                "auc": last.get("auc", np.nan),
            })

    summary = pd.DataFrame(rows)
    _save_df(summary, out / f"scalability_summary_N{len(ctx.clients)}_alpha{ctx.args.dirichlet_alpha}.csv")
    return summary


# -----------------------------------------------------------------------------
# Runtime context and CLI
# -----------------------------------------------------------------------------

class RuntimeContext:
    pass


def build_context(args):
    data_cfg = DataConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        n_qubits=args.n_qubits,
        values_already_angles=args.values_already_angles,
        train_subset_size=args.train_subset_size,
        test_subset_size=args.test_subset_size,
        subset_seed=FINAL_SUBSET_SEED,
        stratify_by_client=not (args.num_clients and args.num_clients > 0),
    )
    qnn_cfg = QNNConfig(
        n_qubits=args.n_qubits,
        ansatz_layers=args.ansatz_layers,
    )
    spsa_cfg = SPSAConfig(
        local_iter=args.local_iter,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        batch_size=args.batch_size,
        max_step_norm=args.max_step_norm,
    )
    pred_cfg = PredictiveConfig(
        warmup_rounds=args.warmup_rounds,
        ema_rho=args.ema_rho,
        residual_small=args.residual_small,
        residual_large=args.residual_large,
        full_update_period=args.full_update_period,
        prediction_shrink=args.prediction_shrink,
        residual_topk_frac=args.residual_topk_frac,
    )
    fed_cfg = FederatedConfig(
        rounds=args.rounds,
        seed=args.seed,
    )
    comm_cfg = CommunicationCost(
        full_value_cost=32.0,
        residual_value_cost=16.0,
        metadata_cost=64.0,
        teleport_ebit_cost=1.0,
        teleport_fidelity=0.95,
        teleport_noise_scale=0.02,
    )

    X_train, X_test, y_train, y_test, clients, info = load_processed_Vehical_Data(data_cfg)

    if args.num_clients and args.num_clients > 0:
        clients = make_dirichlet_clients(
            X_train, y_train,
            n_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            min_client_size=args.min_client_size,
        )
        info["synthetic_partition"] = True
        info["n_clients"] = len(clients)
        info["dirichlet_alpha"] = float(args.dirichlet_alpha)
        info["client_sizes"] = {str(cid): int(len(yc)) for cid, _, yc in clients}
        info["client_pos_rates"] = {str(cid): float(np.mean(yc)) for cid, _, yc in clients}
    else:
        info["synthetic_partition"] = False
        info["n_clients"] = len(clients)

    qnn = StatevectorQNN(qnn_cfg)

    ctx = RuntimeContext()
    ctx.args = args
    ctx.data_cfg = data_cfg
    ctx.qnn_cfg = qnn_cfg
    ctx.spsa_cfg = spsa_cfg
    ctx.pred_cfg = pred_cfg
    ctx.fed_cfg = fed_cfg
    ctx.comm_cfg = comm_cfg
    ctx.X_train = X_train
    ctx.X_test = X_test
    ctx.y_train = y_train
    ctx.y_test = y_test
    ctx.clients = clients
    ctx.data_info = info
    ctx.qnn = qnn

    ctx.out_dir = args.out_dir
    ctx.method = args.method
    ctx.dataset_name = args.dataset_name
    ctx.secret_source = args.secret_source
    ctx.mlkem_algorithm = args.mlkem_algorithm
    ctx.trace_csv = args.trace_csv
    ctx.trace_client_col = args.trace_client_col
    ctx.trace_time_col = args.trace_time_col
    ctx.round_duration_ms = args.round_duration_ms
    ctx.trace_methods = [x.strip() for x in args.trace_methods.split(",") if x.strip()]
    return ctx


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified reviewer-revision vehicular QFL experiments"
    )
    p.add_argument("--stage", default="6",
                   choices=["4","5","6","7","8","9","10","11","12","all"])

    p.add_argument("--train-csv", default="")
    p.add_argument("--test-csv", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", default="HDSVT")

    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--ansatz-layers", type=int, default=2)
    p.add_argument("--values-already-angles", action="store_true")
    p.add_argument("--train-subset-size", type=int, default=5000)
    p.add_argument("--test-subset-size", type=int, default=1500)

    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-clients",
        type=int,
        default=0,
        help="0 keeps the original client_id partition; >0 creates this many synthetic clients."
    )
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration used for synthetic non-IID client partitioning."
    )
    p.add_argument(
        "--min-client-size",
        type=int,
        default=20,
        help="Minimum number of samples allowed in each synthetic client."
    )

    p.add_argument(
        "--mdqfl-clusters",
        type=int,
        default=0,
        help="mdQFL clusters; 0 uses ceil(sqrt(num_clients))."
    )
    p.add_argument(
        "--resfed-topk-frac",
        type=float,
        default=0.25,
        help="Fixed residual Top-K fraction for ResFed-style QFL."
    )
    p.add_argument(
        "--resfed-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Residual value quantization bits for ResFed-style QFL."
    )

    p.add_argument("--local-iter", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--perturbation", type=float, default=0.12)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-step-norm", type=float, default=0.35)

    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--ema-rho", type=float, default=0.70)
    p.add_argument("--residual-small", type=float, default=0.06)
    p.add_argument("--residual-large", type=float, default=0.35)
    p.add_argument("--full-update-period", type=int, default=4)
    p.add_argument("--prediction-shrink", type=float, default=0.75)
    p.add_argument("--residual-topk-frac", type=float, default=0.50)

    p.add_argument("--include-telesync", action="store_true",
                   help="Opt in to the computationally expensive Qiskit/Aer TeleSync-style baseline in Stage 9.")

    p.add_argument(
        "--stage9-literature-only", action="store_true",
        help="For Stage 9, run only mdQFL-style and ResFed-style baselines."
    )
    p.add_argument(
        "--include-literature-baselines", action="store_true",
        help="Include mdQFL-style and ResFed-style baselines in a full Stage-9 run."
    )

    p.add_argument("--method", default="aggregation_impact_qfl",
                   choices=[
                       "tuned_qfl",
                       "predictive_residual_qfl",
                       "aggregation_impact_qfl",
                       "telesync_style_qfl",
                   ])

    p.add_argument("--secret-source", default="mlkem_python",
                   choices=["prototype", "mlkem_python"])
    p.add_argument("--mlkem-algorithm", default="ML-KEM-768",
                   choices=["ML-KEM-512","ML-KEM-768","ML-KEM-1024"])

    # Trace-driven V2X args
    p.add_argument("--trace-csv", default="")
    p.add_argument("--trace-client-col", default="client_id")
    p.add_argument("--trace-time-col", default="timestamp")
    p.add_argument("--round-duration-ms", type=float, default=1000.0)
    p.add_argument(
        "--trace-methods",
        default="tuned_qfl,predictive_residual_qfl,aggregation_impact_qfl"
    )
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Stage 12 only consolidates existing CSVs and does not require loading QNN data.
    if args.stage == "12":
        ctx = RuntimeContext()
        ctx.out_dir = args.out_dir
        stage12_consolidate(ctx)
        return

    if not args.train_csv or not args.test_csv:
        raise ValueError("--train-csv and --test-csv are required for stages 4-11 and 'all'.")

    ctx = build_context(args)
    print("[config]", json.dumps({
        "stage": args.stage,
        "dataset": args.dataset_name,
        "rounds": args.rounds,
        "seed": args.seed,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "out_dir": args.out_dir,
        "trace_csv": args.trace_csv,
        "frozen_protocol": FINAL_STAGE9_PROTOCOL,
        "subset_seed": FINAL_SUBSET_SEED,
    }, indent=2), flush=True)

    if args.stage in {"4", "all"}:
        stage4_scalability(ctx)
    if args.stage in {"5", "all"}:
        stage5_security_profiles(ctx)
    if args.stage in {"6", "all"}:
        stage6_security_ablation(ctx)
    if args.stage in {"7", "all"}:
        stage7_topk(ctx)
    if args.stage in {"8", "all"}:
        stage8_topk_error_feedback(ctx)
    if args.stage in {"9", "all"}:
        if getattr(args, "stage9_literature_only", False):
            stage9_literature_baselines_only(ctx)
        else:
            stage9_quantization_and_reference(ctx)
    if args.stage in {"10", "all"}:
        stage10_trace_v2x(ctx)
    if args.stage in {"11", "all"}:
        stage11_handover_staleness(ctx)
    if args.stage == "all":
        stage12_consolidate(ctx)


if __name__ == "__main__":
    main()
