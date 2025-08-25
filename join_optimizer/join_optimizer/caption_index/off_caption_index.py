import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.fts_store.db_fts_store import SQLiteFTSStore
from lotus.models.instructblip_cm import InstructBlipCaptioner

load_dotenv()

OFF_DATASET_DIR = os.getenv("OFF_DATASET_DIR")

OFF_PARQUET = os.path.join(OFF_DATASET_DIR, "products.parquet")
OFF_IMAGES_DIR = os.path.join(OFF_DATASET_DIR, "images")
INDEX_DB = "off_uk_top2000_with_images_caps_instructblip-flan-t5-xl.db"

assert os.path.exists(OFF_PARQUET), f"Missing parquet: {OFF_PARQUET}"
assert os.path.isdir(OFF_IMAGES_DIR), f"Missing images dir: {OFF_IMAGES_DIR}"

codes = duckdb.query(f"""
    SELECT code
    FROM parquet_scan('{OFF_PARQUET}')
    ORDER BY code ASC
""").to_df()

df = pd.DataFrame({"image_path": codes["code"].apply(lambda c: os.path.join(OFF_IMAGES_DIR, f"{c}.jpg"))})
df["image"] = ImageArray(df["image_path"])
df = df[["image"]]

lotus.settings.configure(cm=InstructBlipCaptioner(), cs=SQLiteFTSStore())
df = df.sem_captions_index("image", index_dir=INDEX_DB, batch_size=16)

print(df.head())
print(f"Index stored at: {INDEX_DB}")
