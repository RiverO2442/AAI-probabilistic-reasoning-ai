#############################################################################
# ModelEvaluator.py (Final, Optimized Version)
#
# Evaluates Bayesian Network models using probabilistic inference.
# Supports binary classification (e.g., heart disease, fraud detection).
#
# Implements:
#  - Balanced Accuracy, F1 Score, AUC, Brier Score
#  - Kullback-Leibler Divergence (KL Div)
#  - Expected Calibration Loss (ECL)
#  - Inference time (sec)
#
# Compatible with: BayesNetInference, BayesNetReader, BayesNetUtil
# Author: Hao Nguyen (MSc Computer Science, University of Lincoln)
# Updated: November 2025
#############################################################################

import sys
import math
import time
import numpy as np
from sklearn import metrics, calibration
import BayesNetUtil as bnu
from DataReader import CSV_DataReader
from BayesNetInference import BayesNetInference


class ModelEvaluator(BayesNetInference):
    verbose = False
    inference_time = None

    def __init__(self, configfile_name, datafile_test):
        # --- Load Bayesian Network (no inference at init) ---
        super().__init__(configfile_name, None)

        # --- Load test dataset ---
        self.csv = CSV_DataReader(datafile_test)

        # --- Automatically detect or override target variable ---
        possible_targets = ["target", "Fraud_Label", "label", "class"]
        for t in possible_targets:
            if t in self.csv.rand_vars:
                self.csv.predictor_variable = t
                break
        else:
            self.csv.predictor_variable = self.csv.rand_vars[-1]

        print(f"Predictor variable detected: {self.csv.predictor_variable}")

        # --- Run inference and evaluate performance ---
        self.inference_time = time.time()
        true, pred, prob = self.get_true_and_predicted_targets()
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(true, pred, prob)

    # ======================================================================
    # MAIN INFERENCE + EVALUATION LOGIC
    # ======================================================================

    def get_true_and_predicted_targets(self):
        print("\nCARRYING OUT probabilistic inference on test data...")
        Y_true, Y_pred, Y_prob = [], [], []

        target_col_index = self.csv.rand_vars.index(self.csv.predictor_variable)

        for i, data_point in enumerate(self.csv.rv_all_values):
            target_value = data_point[target_col_index]

            # --- Convert labels to binary (0/1) ---
            if str(target_value).lower() in ["yes", "1", "true"]:
                Y_true.append(1)
            elif str(target_value).lower() in ["no", "0", "false"]:
                Y_true.append(0)
            else:
                continue  # skip if unrecognized label

            # --- Obtain probabilistic predictions ---
            prob_dist = self.get_predictions_from_BayesNet(data_point)

            # --- Extract probability of the positive class (1) ---
            try:
                predicted_output = prob_dist["1"]
            except Exception:
                predicted_output = list(prob_dist.values())[0]

            if str(target_value) in ["no", "0", "false"]:
                predicted_output = 1 - predicted_output

            Y_prob.append(predicted_output)

            # --- Hard prediction (most probable label) ---
            best_key = max(prob_dist, key=prob_dist.get)
            if str(best_key).lower() in ["yes", "1", "true"]:
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        # --- Sanity check ---
        if len(Y_true) == 0:
            print("⚠️ WARNING: No valid target values found in test data.")
        if len(Y_true) != len(Y_prob):
            print(f"⚠️ WARNING: Mismatch Y_true={len(Y_true)} vs Y_prob={len(Y_prob)}")

        return Y_true, Y_pred, Y_prob

    # ======================================================================
    # BAYESIAN INFERENCE WRAPPER
    # ======================================================================

    def get_predictions_from_BayesNet(self, data_point):
        # Build evidence from all non-target variables
        evidence_items = []
        for var_index in range(0, len(self.csv.rand_vars)):
            var_name = self.csv.rand_vars[var_index]
            var_value = str(data_point[var_index])
            if (
                var_name != self.csv.predictor_variable
                and var_name in self.bn["random_variables"]
            ):
                evidence_items.append(f"{var_name}={var_value}")

        evidence = ",".join(evidence_items)
        prob_query = f"P({self.csv.predictor_variable}|{evidence})"

        # Tokenise and run inference
        self.query = bnu.tokenise_query(prob_query)
        try:
            self.prob_dist = self.enumeration_ask()
            normalised_dist = bnu.normalise(self.prob_dist)
        except KeyError as e:
            print(f"[WARNING] Missing variable in evidence: {e}. Skipping instance.")
            return {"0": 0.5, "1": 0.5}
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return {"0": 0.5, "1": 0.5}

        if self.verbose:
            print(f"{prob_query} = {normalised_dist}")

        return normalised_dist

    # ======================================================================
    # PERFORMANCE METRICS
    # ======================================================================

    def expected_calibration_loss(self, y_true, y_prob, n_bins=None):
        if n_bins is None:
            N = len(y_true)
            n_bins = max(5, math.ceil(math.log2(N) + 1))
        prob_true, prob_pred = calibration.calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        nonempty = bin_counts > 0
        bin_weights = bin_counts[nonempty] / np.sum(bin_counts[nonempty])
        return np.sum(bin_weights * np.abs(prob_true - prob_pred))

    def compute_performance(self, Y_true, Y_pred, Y_prob):
        if len(Y_true) == 0:
            print("❌ No valid predictions; cannot compute performance metrics.")
            return

        P = np.asarray(Y_true) + 1e-8
        Q = np.asarray(Y_prob) + 1e-8

        print("\nCOMPUTING PERFORMANCE METRICS...")
        try:
            bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
            f1 = metrics.f1_score(Y_true, Y_pred)
            fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            brier = metrics.brier_score_loss(Y_true, Y_prob)
            kl_div = np.sum(P * np.log(P / Q))
            ec_loss = self.expected_calibration_loss(Y_true, Y_prob)

            print(f"Balanced Accuracy = {bal_acc:.4f}")
            print(f"F1 Score          = {f1:.4f}")
            print(f"AUC               = {auc:.4f}")
            print(f"Brier Score       = {brier:.4f}")
            print(f"KL Divergence     = {kl_div:.4f}")
            print(f"Expected Cal Loss = {ec_loss:.4f}")
            print(f"Inference Time    = {self.inference_time:.4f} secs")

        except Exception as e:
            print(f"[ERROR] Metric computation failed: {e}")


# ======================================================================
# MAIN EXECUTION (CLI)
# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-heart.txt heart_test.csv")
        sys.exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        ModelEvaluator(configfile, datafile_test)
