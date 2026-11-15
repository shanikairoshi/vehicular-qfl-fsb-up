# Modeling Secure and Communication-Efficient Quantum Federated Learning for Vehicular Networks

**Authors:** Shanika I. Nanayakkara and Shiva Raj Pokhrel  
**Manuscript:** IEEE Transactions on Mobile Computing (under revision)

## Overview

This repository accompanies a study of secure and communication-efficient quantum federated learning (QFL) for vehicular networks. The framework combines adaptive angular-update communication with federated-semantics-bound update protection (FSB-UP), and evaluates learning utility, modeled communication cost, security and mobility-related impairments.

## Methods

- **PR-QFL:** Predictive-residual QFL selects communication modes according to prediction discrepancy and residual reconstruction.
- **AI-QFL:** Aggregation-impact QFL selects communication modes using estimated aggregation impact.
- **FSB-UP:** Application-layer update protection using ML-KEM shared-secret establishment, HKDF context binding, authenticated encryption, expected-state validation and replay protection.

The communication comparison includes FP32, INT8, INT4, Top-K, Top-K with error feedback, mdQFL-style, ResFed-style, PR-QFL and AI-QFL. Literature-inspired implementations are adaptations, not claims of exact reproduction of their original systems.

## Repository organization

The intended release contains the frozen experimental implementation (`qfl_revision/`), preprocessing notebooks (`preprocessing/`), statistical analysis notebooks (`analysis/`), archived result CSVs (`results/`), publication figures (`figures/`) and supporting documentation (`docs/`). See [reproducibility checklist](docs/REPRODUCIBILITY.md) and [experiment mapping](docs/EXPERIMENT_MAPPING.md).

## Experimental configuration

The principal communication comparison uses 20 clients, 100 rounds, Dirichlet concentration 0.5 and seeds 42, 123 and 2024. The trace-driven V2X study uses 20 clients, 30 rounds and the same three seeds. The reconstruction diagnostic uses 10 clients, 20 rounds and seed 42. These are distinct experiments; consult the corresponding frozen configurations before execution.

## Data availability

The experiments use HDSVT, NGSIM and OpenDD. See [dataset documentation](docs/DATASETS.md). Raw datasets are not included by default; access and redistribution are subject to their original terms.

## Reproducibility

Two workflows are intended: (1) rerun the frozen experiments from prepared datasets and documented configurations; (2) regenerate reported figures and tables from archived per-run CSVs without repeating quantum simulations. Before running analysis notebooks, configure their input and output paths for the local checkout. Exact commands, dependency versions and artifact paths must be verified against the frozen implementation before declaring the package complete.

## Scope and limitations

Quantum learning is simulator-based. OpenDD trajectories provide mobility inputs, while RSU infrastructure and link impairments are simulated. Communication costs are modeled. FSB-UP assumes externally authenticated credentials and is not a general authenticated key-establishment protocol or a defence against context-valid model poisoning. TeleSync-style transport represents idealized simulator-based resource accounting, not physical quantum-network validation; its delay and handover statistics are not directly comparable pending event-accounting validation.

## Citation

Shanika I. Nanayakkara and Shiva Raj Pokhrel, “Modeling Secure and Communication-Efficient Quantum Federated Learning for Vehicular Networks,” manuscript under revision for IEEE Transactions on Mobile Computing.

## Release status

This documentation is a preparation draft. File availability, execution commands, numerical consistency, licensing and public release authorization must be checked before publication.
