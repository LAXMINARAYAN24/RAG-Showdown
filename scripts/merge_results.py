# scripts/merge_results.py
import pandas as pd
import glob

files = glob.glob("eval/results/eval_*.csv")
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

combined.to_csv("eval/results/combined_all_strategies.csv", index=False)
print(f"Merged {len(files)} files into eval/results/combined_all_strategies.csv")
print(combined.groupby("strategy")["score"].mean())