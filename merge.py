import numpy as np
from pathlib import Path

REQUIRED_KEYS = ["states", "policies", "values", "board_size", "action_size"]
out_path = Path("combined_all.npz")
search_root = Path("..").resolve()


def discover_sources(root: Path, output_file: Path):
    npz_sources = sorted(
        p for p in root.glob("*.npz")
        if p.is_file() and p.name != output_file.name
    )
    if npz_sources:
        return [str(p) for p in npz_sources]

    dir_sources = sorted(
        p for p in root.iterdir()
        if p.is_dir() and all((p / f"{k}.npy").is_file() for k in REQUIRED_KEYS)
    )
    if dir_sources:
        return [str(p) for p in dir_sources]

    raise ValueError(
        f"No sources found in {root.resolve()}. "
        "Expected .npz files or folders with states.npy, policies.npy, values.npy, board_size.npy, action_size.npy."
    )

def load_source(src: str):
    p = Path(src)

    if p.is_file() and p.suffix == ".npz":
        d = np.load(p, allow_pickle=False)
        missing = [k for k in REQUIRED_KEYS if k not in d.files]
        if missing:
            raise ValueError(f"{p} is missing keys: {missing}")
        item = {k: d[k] for k in REQUIRED_KEYS}
        return item, f"npz:{p.name}"

    if p.is_dir():
        # folder containing .npy files
        item = {}
        for k in REQUIRED_KEYS:
            f = p / f"{k}.npy"
            if not f.exists():
                raise ValueError(f"Folder {p} is missing {f.name}")
            item[k] = np.load(f, allow_pickle=False)
        return item, f"dir:{p.name}"

    raise ValueError(f"Source not found or unsupported: {src}")

loaded = []
sources = discover_sources(search_root, out_path)
print("Using sources:", sources)

for s in sources:
    item, label = load_source(s)
    loaded.append((item, label))

# ---- Validate compatibility across all sources ----
board_size0 = int(np.asarray(loaded[0][0]["board_size"]).item())
action_size0 = int(np.asarray(loaded[0][0]["action_size"]).item())

state_shape0   = loaded[0][0]["states"].shape[1:]
policy_shape0  = loaded[0][0]["policies"].shape[1:]
value_shape0   = loaded[0][0]["values"].shape[1:]

for item, label in loaded:
    bs = int(np.asarray(item["board_size"]).item())
    ac = int(np.asarray(item["action_size"]).item())

    if bs != board_size0:
        raise ValueError(f"{label}: board_size mismatch {bs} != {board_size0}")
    if ac != action_size0:
        raise ValueError(f"{label}: action_size mismatch {ac} != {action_size0}")
    if item["states"].shape[1:] != state_shape0:
        raise ValueError(f"{label}: states shape mismatch {item['states'].shape[1:]} != {state_shape0}")
    if item["policies"].shape[1:] != policy_shape0:
        raise ValueError(f"{label}: policies shape mismatch {item['policies'].shape[1:]} != {policy_shape0}")
    if item["values"].shape[1:] != value_shape0:
        raise ValueError(f"{label}: values shape mismatch {item['values'].shape[1:]} != {value_shape0}")

# ---- Concatenate ----
states   = np.concatenate([item["states"]   for item, _ in loaded], axis=0).astype(np.float32, copy=False)
policies = np.concatenate([item["policies"] for item, _ in loaded], axis=0).astype(np.float32, copy=False)
values   = np.concatenate([item["values"]   for item, _ in loaded], axis=0).astype(np.float32, copy=False)

np.savez_compressed(
    out_path,
    states=states,
    policies=policies,
    values=values,
    board_size=np.asarray(board_size0, dtype=np.int32),
    action_size=np.asarray(action_size0, dtype=np.int32),
)

print("Saved combined dataset:", out_path)
print("states:", states.shape, states.dtype)
print("policies:", policies.shape, policies.dtype)
print("values:", values.shape, values.dtype)
print("board_size:", board_size0, "action_size:", action_size0)
