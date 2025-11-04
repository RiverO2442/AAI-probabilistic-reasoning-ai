#!/usr/bin/env python3
"""
train_and_evaluate_heart.py

End-to-end script to:
 - discretize heart disease CSV,
 - run 5-fold stratified CV:
    * learn BN structure (bnlearn) on TRAIN
    * write config file (variables + structure)
    * generate CPTs from TRAIN (CPT_Generator)
    * evaluate on TEST (ModelEvaluator -- prints metrics)

Usage:
    python train_and_evaluate_heart.py \
        --raw_csv data/data_heart.csv \
        --out_dir experiments/heart_cv \
        --kfolds 5
"""

import os
import argparse
import pandas as pd
import numpy as np
import bnlearn as bn
import shutil
import subprocess
import tempfile
from sklearn.model_selection import StratifiedKFold

# Change these to match your filenames/locations if different
DEFAULT_RAW_CSV = "data/heart.csv"

# Discretization settings (3 bins -> labels low/medium/high)
CONTINUOUS_BINS = {
    'age': [0, 40, 60, 120],
    'trestbps': [0, 120, 140, 300],
    'chol': [0, 200, 240, 600],
    'thalach': [0, 120, 160, 250],
    'oldpeak': [0, 1, 2, 10]
}
BIN_LABELS = ['low', 'medium', 'high']

def discretize_dataframe(df):
    df2 = df.copy()
    for col, bins in CONTINUOUS_BINS.items():
        if col in df2.columns:
            df2[f"{col}_bin"] = pd.cut(df2[col], bins=bins, labels=BIN_LABELS, include_lowest=True)
            # convert categorical labels to strings (CPT_Generator expects strings)
            df2[f"{col}_bin"] = df2[f"{col}_bin"].astype(str)
    return df2

def write_config_file(config_path, name, rand_vars, parent_map):
    """
    Writes a config file compatible with your BayesNetReader format.

    - rand_vars: list of variable names (strings) in desired order
    - parent_map: dict var -> list of parent var names (may be empty)
    """
    # Remove any parents not in current random variables (avoid missing columns like oldpeak)
    for var, parents in parent_map.items():
        parent_map[var] = [p for p in parents if p in rand_vars]
    # ensure format: random_variables separated by semicolon
    rand_vars_str = ';'.join(rand_vars)
    # Build structure tokens like P(var|p1,p2); or P(var)
    structure_entries = []
    for var in rand_vars:
        parents = parent_map.get(var, [])
        if parents:
            parents_str = ','.join(parents)
            structure_entries.append(f"P({var}|{parents_str})")
        else:
            structure_entries.append(f"P({var})")
    structure_str = ';'.join(structure_entries)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"name:{name}\n\n")
        f.write(f"random_variables:{rand_vars_str}\n\n")
        f.write(f"structure:{structure_str}\n\n")
    print(f"[CONFIG] wrote config to {config_path}")

def learn_structure_and_parents(train_df, rand_vars):
    """
    Learn BN structure using bnlearn. Return parent map: var -> [parents].
    """
    print("[STRUCTURE] learning BN structure with bnlearn (hillclimb + BIC)...")
    model = bn.structure_learning.fit(train_df, methodtype='hillclimbsearch', scoretype='bic', verbose=False)
    edges = model.get('model_edges', [])
    # edges format from bnlearn is usually list of tuples (u,v) meaning u -> v
    # Build parent map
    parent_map = {v: [] for v in rand_vars}
    for e in edges:
        if len(e) == 2:
            u, v = e
            # ensure u and v are string names and in variables
            if v in parent_map:
                parent_map[v].append(u)
            else:
                # if v not in rand_vars, still add to map
                parent_map.setdefault(v, []).append(u)
    return parent_map, edges

def run_cpt_generator(config_path, train_csv_path):
    """
    Invoke CPT_Generator class by importing if possible, else run as subprocess.
    We will attempt to import CPT_Generator; if that fails, call via subprocess.
    """
    try:
        # Try importing module and instantiating class
        from CPT_Generator import CPT_Generator
        # CPT_Generator constructor runs the process
        CPT_Generator(config_path, train_csv_path)
        return True
    except Exception as e:
        print("[CPT_GENERATOR] import approach failed:", e)
        # fallback to running as a subprocess (requires CPT_Generator.py to be executable)
        try:
            cmd = ["python", "CPT_Generator.py", config_path, train_csv_path]
            print("[CPT_GENERATOR] running subprocess:", ' '.join(cmd))
            subprocess.check_call(cmd)
            return True
        except subprocess.CalledProcessError as se:
            print("[CPT_GENERATOR] subprocess failed:", se)
            return False

def run_model_evaluator(config_path, test_csv_path):
    """
    Run ModelEvaluator.py as subprocess and capture stdout.
    Returns the captured stdout text.
    """
    cmd = ["python", "ModelEvaluator.py", config_path, test_csv_path]
    print("[EVALUATOR] running:", ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    return out

def prepare_rand_vars_list(df):
    """
    Prepare the list of random variables for config file.
    We include discretized bin columns instead of original numeric columns.
    Keep original categorical columns as-is and ensure target is last.
    """
    cols = list(df.columns)
    # keep order but if any continuous original column exists, replace it with its '_bin' version
    result = []
    for c in cols:
        if c in CONTINUOUS_BINS:
            # replace with c_bin
            bincol = f"{c}_bin"
            if bincol in df.columns:
                result.append(bincol)
        else:
            result.append(c)
    # ensure uniqueness and keep order
    seen = set()
    final = []
    for x in result:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final

def main(raw_csv, out_dir, kfolds):
    os.makedirs(out_dir, exist_ok=True)
    df_raw = pd.read_csv(raw_csv)
    print("[DATA] loaded raw CSV shape:", df_raw.shape)

    # Discretize continuous columns
    df_disc = discretize_dataframe(df_raw)
    print("[DATA] after discretization, columns:", df_disc.columns.tolist())

    # Prepare variables list (target must be last)
    # We assume last column in raw CSV is target by convention; preserve that
    target_col = df_raw.columns[-1]
    if target_col not in df_disc.columns:
        raise ValueError("Target column not found after discretization.")
    rand_vars = prepare_rand_vars_list(df_disc)
    # If target isn't last in rand_vars, move it to last
    if rand_vars[-1] != target_col:
        rand_vars = [c for c in rand_vars if c != target_col] + [target_col]

    # Stratified K-Fold
    y = df_disc[target_col].astype(str).values  # ensure string labels for stratified split
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

    fold = 0
    summary_outputs = []
    for train_idx, test_idx in skf.split(df_disc, y):
        fold += 1
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df_disc.iloc[train_idx].reset_index(drop=True)
        test_df = df_disc.iloc[test_idx].reset_index(drop=True)

        train_csv = os.path.join(fold_dir, "train.csv")
        test_csv = os.path.join(fold_dir, "test.csv")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"\n[CV] Fold {fold} - Train shape: {train_df.shape} Test shape: {test_df.shape}")

        # Learn structure on training data (discrete)
        parent_map, edges = learn_structure_and_parents(train_df, rand_vars)
        print(f"[CV] Fold {fold} - learned edges count:", len(edges))

        # Write temporary config file for this fold
        config_path = os.path.join(fold_dir, "config_heart.txt")
        config_name = f"HeartFold{fold}"
        write_config_file(config_path, config_name, rand_vars, parent_map)

        # Run CPT_Generator to compute CPTs from training data and update config
        ok = run_cpt_generator(config_path, train_csv)
        if not ok:
            print("[ERROR] CPT generation failed for fold", fold)
            continue

        # Run ModelEvaluator on this config and test CSV. Capture stdout.
        evaluator_output = run_model_evaluator(config_path, test_csv)
        print(f"[CV] Fold {fold} evaluator output:\n{evaluator_output}")
        summary_outputs.append((fold, evaluator_output))

    # Print summarized outputs
    print("\n\n===== Cross-validation Summary =====")
    for fold, out in summary_outputs:
        print(f"--- Fold {fold} ---")
        print(out)
        print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate BN model for Heart dataset (discrete) with K-fold CV.")
    parser.add_argument("--raw_csv", type=str, default=DEFAULT_RAW_CSV, help="Path to raw heart CSV")
    parser.add_argument("--out_dir", type=str, default="experiments/heart_cv", help="Output directory to store folds/configs")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of stratified folds (default=5)")

    args = parser.parse_args()
    main(args.raw_csv, args.out_dir, args.kfolds)
