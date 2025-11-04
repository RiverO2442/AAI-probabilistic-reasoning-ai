import bnlearn as bn
import pandas as pd
from bnlearn_ConditionalIndependenceTests import save_structure

# Load preprocessed fraud data
df = pd.read_csv("./data/fraud_data_processed.csv")

# Learn structure (discrete)
model = bn.structure_learning.fit(
    df,
    methodtype='hillclimbsearch',  # fast + accurate for tabular data
    scoretype='bic',
    max_iter=10000
)
print("Learned edges:", model['model_edges'])

# Save the DAG image
save_structure(model['model_edges'], "Fraud Detection Structure", "structures/fraud_DAG.png")

# Save structure for config
edges = model['model_edges']
rand_vars = list(df.columns)
print("✅ Structure learned for fraud dataset.")
