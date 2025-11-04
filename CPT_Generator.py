#############################################################################
# CPT_Generator.py
#
# Generates Conditional Probability Tables (CPTs) into a config file for
# probabilistic inference using a Bayesian Network.
#
# This version supports:
# - Laplacian smoothing (l=1)
# - Automatic data balancing via SMOTE (for imbalanced datasets)
#
# Usage:
#   python CPT_Generator.py ./config/config_fraud.txt ./data/fraud_data_processed.csv
#
# Version: 2.0, Date: 04 November 2025
# Author: Adapted for Advanced AI Assessment
#############################################################################

import sys
from BayesNetReader import BayesNetReader
from DataReader import CSV_DataReader
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

class CPT_Generator(BayesNetReader):
    def __init__(self, configfile_name, datafile_name):
        self.configfile_name = configfile_name
        print(f"\n📁 Loading and balancing data from {datafile_name}...")

        # --- Step 1: Load data ---
        df = pd.read_csv(datafile_name)

        # Automatically encode categorical features
        print("🔢 Encoding categorical columns before balancing...")
        for col in df.columns:
            if df[col].dtype == 'object':
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                print(f"Encoded '{col}' -> {len(encoder.classes_)} unique values")

        # Automatically balance data if Fraud_Label exists
        if "Fraud_Label" in df.columns:
            print("🔄 Detected 'Fraud_Label' — performing SMOTE balancing...")
            X = df.drop(columns=["Fraud_Label"])
            y = df["Fraud_Label"]

            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            df_balanced = pd.concat([X_res, y_res], axis=1)
            balanced_path = "./data/fraud_data_balanced.csv"
            os.makedirs(os.path.dirname(balanced_path), exist_ok=True)
            df_balanced.to_csv(balanced_path, index=False)
            print(f"✅ Balanced dataset created at {balanced_path}")
            print("Class distribution:", Counter(y_res))
            datafile_name = balanced_path
        else:
            print("ℹ️ No Fraud_Label column found — using original data.")

        # --- Step 2: Read Bayes Net and Data ---
        self.bn = BayesNetReader(configfile_name)
        self.csv = CSV_DataReader(datafile_name)

        # --- Step 3: Generate CPTs ---
        self.constant_l = 1.0  # Laplacian smoothing constant
        self.countings = {}
        self.CPTs = {}
        self.generate_prior_and_conditional_countings()
        self.generate_probabilities_from_countings()
        self.write_CPTs_to_configuration_file()

    # ---------------------------------------------------------------
    # Step 4: Generate countings
    # ---------------------------------------------------------------
    def generate_prior_and_conditional_countings(self):
        print("\n📊 GENERATING countings for prior/conditional distributions...")
        print("-------------------------------------------------------------")

        for pd_str in self.bn.bn["structure"]:
            print(f"Processing: {pd_str}")
            p = pd_str.replace("(", " ").replace(")", " ")
            tokens = p.split("|")

            # Prior probabilities
            if len(tokens) == 1:
                variable = tokens[0].split(" ")[1]
                variable_index = self.get_variable_index(variable)
                counts = self.initialise_counts(variable)
                self.get_counts(variable_index, None, counts)

            # Conditional probabilities
            elif len(tokens) == 2:
                variable = tokens[0].split(" ")[1]
                variable_index = self.get_variable_index(variable)
                parents = tokens[1].strip().split(",")
                parent_indexes = self.get_parent_indexes(parents)
                counts = self.initialise_counts(variable, parents)
                self.get_counts(variable_index, parent_indexes, counts)

            else:
                print(f"⚠️ Unexpected structure format: {pd_str}")
                continue

            self.countings[pd_str] = counts
            print("Counts:", counts, "\n")

    # ---------------------------------------------------------------
    # Step 5: Convert countings into probabilities
    # ---------------------------------------------------------------
    def generate_probabilities_from_countings(self):
        print("\n🧮 GENERATING prior and conditional probabilities...")
        print("---------------------------------------------------")

        for pd_str, counts in self.countings.items():
            print(f"Generating probabilities for {pd_str}")
            tokens = pd_str.split("|")
            variable = tokens[0].replace("P(", "")
            cpt = {}

            # --- Case 1: Prior ---
            if len(tokens) == 1:
                total = sum(counts.values())
                Jl = len(counts) * self.constant_l
                for key, count in counts.items():
                    cpt[key] = (count + self.constant_l) / (total + Jl)

            # --- Case 2: Conditional ---
            elif len(tokens) == 2:
                parents_values = self.get_parent_values(counts)
                for parents_value in parents_values:
                    total = sum(
                        count for key, count in counts.items() if key.endswith("|" + parents_value)
                    )
                    J = len(self.csv.rv_key_values[variable])
                    Jl = J * self.constant_l
                    for key, count in counts.items():
                        if key.endswith("|" + parents_value):
                            cpt[key] = (count + self.constant_l) / (total + Jl)

            self.CPTs[pd_str] = cpt
            print("CPT:", cpt, "\n")

    # ---------------------------------------------------------------
    # Step 6: Helpers for variable lookup and counting
    # ---------------------------------------------------------------
    def get_variable_index(self, variable):
        for i, var in enumerate(self.csv.rand_vars):
            if variable == var:
                return i
        print(f"⚠️ Variable index not found: {variable}")
        return None

    def get_parent_indexes(self, parents):
        return [self.get_variable_index(p) for p in parents]

    def get_parent_values(self, counts):
        values = []
        for key in counts.keys():
            if "|" in key:
                val = key.split("|")[1]
                if val not in values:
                    values.append(val)
        return values

    def initialise_counts(self, variable, parents=None):
        counts = {}
        if parents is None:
            for val in self.csv.rv_key_values[variable]:
                counts[val] = 0
        else:
            # Enumerate all combinations of parent variable values
            parent_combinations = [""]
            for parent in parents:
                new_combos = []
                for combo in parent_combinations:
                    for val in self.csv.rv_key_values[parent]:
                        new_combos.append((combo + "," + val).strip(","))
                parent_combinations = new_combos
            for val in self.csv.rv_key_values[variable]:
                for combo in parent_combinations:
                    counts[val + "|" + combo] = 0
        return counts

    def get_counts(self, var_index, parent_indexes, counts):
        for values in self.csv.rv_all_values:
            if parent_indexes is None:
                value = values[var_index]
            else:
                parent_vals = ",".join([values[i] for i in parent_indexes])
                value = values[var_index] + "|" + parent_vals
            if value in counts:
                counts[value] += 1

    # ---------------------------------------------------------------
    # Step 7: Write CPTs back to configuration file
    # ---------------------------------------------------------------
    def write_CPTs_to_configuration_file(self):
        print("\n📝 WRITING config file with CPT tables...")
        print("---------------------------------------------------")
        name = self.bn.bn["name"]

        rand_vars = self.bn.bn["random_variables_raw"]
        rand_vars = str(rand_vars).replace("[", "").replace("]", "").replace("'", "").replace(", ", ";")

        structure = self.bn.bn["structure"]
        structure = str(structure).replace("[", "").replace("]", "").replace("'", "").replace(", ", ";")

        with open(self.configfile_name, "w") as cfg_file:
            cfg_file.write(f"name:{name}\n\n")
            cfg_file.write(f"random_variables:{rand_vars}\n\n")
            cfg_file.write(f"structure:{structure}\n\n")

            for key, cpt in self.CPTs.items():
                header = key.replace("P(", "CPT(")
                cfg_file.write(f"{header}:\n")
                for i, (domain_vals, prob) in enumerate(cpt.items()):
                    line = f"{domain_vals}={prob}"
                    if i < len(cpt) - 1:
                        line += ";"
                    cfg_file.write(line + "\n")
                cfg_file.write("\n")

        print(f"✅ CPTs successfully written to {self.configfile_name}")


# ---------------------------------------------------------------
# Run from command line
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: CPT_Generator.py [your_config_file.txt] [training_file.csv]")
        print("EXAMPLE> CPT_Generator.py ./config/config_fraud.txt ./data/fraud_data_processed.csv")
        sys.exit(0)
    else:
        configfile_name = sys.argv[1]
        datafile_name = sys.argv[2]
        CPT_Generator(configfile_name, datafile_name)
