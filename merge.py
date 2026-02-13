import numpy as np
from pathlib import Path

# -------------------- EDIT THESE 4 PATHS --------------------
sources = [
    "250_games.npz",      # .npz
    "mcts_dataset_merged.npz",      # .npz
    "mcts_dataset_merged_1725_seed_0.npz",        # folder with states.npy, policies.npy, values.npy, board_size.npy, action_size.npy
    "mcts_dataset_merged_1000_seed_1.npz",        # folder with the same 5 .npy files
]
merged = Path("combined_all.npz")
# ------------------------------------------------------------

REQUIRED_KEYS = ["states", "policies", "values", "board_size", "action_size"]

def load_policies(src: str):
    p = Path(src)
    if p.is_file() and p.suffix == ".npz":
        d = np.load(p, allow_pickle=False)
        if "policies" not in d.files:
            raise ValueError(f"{p} missing 'policies'")
        return d["policies"], f"npz:{p.name}"

    if p.is_dir():
        f = p / "policies.npy"
        if not f.exists():
            raise ValueError(f"{p} missing policies.npy")
        return np.load(f, allow_pickle=False), f"dir:{p.name}"

    raise ValueError(f"Not found/unsupported: {src}")

def pretty(n): return f"{n:,}"

# Per-source counts
counts = []
sum_rows = 0
print("Per-source policy counts:")
print("-" * 60)
for s in sources:
    pol, label = load_policies(s)
    rows, cols = pol.shape
    counts.append((label, rows, cols))
    sum_rows += rows
    print(f"{label:25s}  policies shape={pol.shape}  rows={pretty(rows)}")

print("-" * 60)
print("Sum of source policy rows:", pretty(sum_rows))

# Merged count
m = np.load(merged, allow_pickle=False)
merged_rows, merged_cols = m["policies"].shape
print("Merged policies shape:", m["policies"].shape, "rows=", pretty(merged_rows))

# Check equals
if merged_rows == sum_rows:
    print("✅ OK: merged rows match sum of sources")
else:
    print("❌ MISMATCH: merged rows != sum of sources")
    print("   Difference:", pretty(merged_rows - sum_rows))