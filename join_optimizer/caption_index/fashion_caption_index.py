import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.fts_store.db_fts_store import SQLiteFTSStore
from lotus.models.blip_cm import BlipCaptioner

load_dotenv()

FASHION_DATASET_DIR = os.getenv("FASHION_DATASET_DIR")

FASHION_PARQUET = os.path.join(FASHION_DATASET_DIR, "styles.parquet")
FASHION_IMAGES_DIR = os.path.join(FASHION_DATASET_DIR, "images")
INDEX_DB = "fashion_dataset_caps_blip-image-captioning-large.db"

ids = duckdb.query(f"""
    SELECT id
    FROM parquet_scan('{FASHION_PARQUET}')
    ORDER BY id
""").to_df()

df = pd.DataFrame({"image_path": ids["id"].apply(lambda i: os.path.join(FASHION_IMAGES_DIR, f"{int(i)}.jpg"))})
df["image"] = ImageArray(df["image_path"])
df = df[["image"]]

lotus.settings.configure(cm=BlipCaptioner(), cs=SQLiteFTSStore())
df = df.sem_captions_index("image", index_dir=INDEX_DB, batch_size=64)

print(df.head())
print(f"Index stored at: {INDEX_DB}")
