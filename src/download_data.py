import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE = "hf://datasets/yandex/yambda/flat/50m"

for name in ["listens", "likes", "dislikes"]:
    print(f"Скачиваю {name}...")
    df = pl.read_parquet(f"{BASE}/{name}.parquet")
    df.write_parquet(DATA_DIR / f"{name}.parquet")
    print(f"{name}: {df.shape}")

print("Готово!")