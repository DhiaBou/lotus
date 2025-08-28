import os

import duckdb
import pandas as pd
from dotenv import load_dotenv

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.fts_store.db_fts_store import SQLiteFTSStore
from lotus.models import LM
from lotus.models.llm_cm import LLMCaptioner

load_dotenv()

OFF_DATASET_DIR = os.getenv("OFF_DATASET_DIR")

OFF_PARQUET = os.path.join(OFF_DATASET_DIR, "products.parquet")
OFF_IMAGES_DIR = os.path.join(OFF_DATASET_DIR, "images")
INDEX_DB = os.path.join(OFF_DATASET_DIR, "llm.db")

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

gpt_5_mini = LM("gpt-5-nano")
cm = LLMCaptioner(
    user_instruction="Describe the image faithfully in plain text without special characters.",
    # user_instruction="Just say A",
    strategy=None,
)

lotus.settings.configure(lm=gpt_5_mini, cm=cm, cs=SQLiteFTSStore())
df = df.sem_captions_index("image", index_dir=INDEX_DB, batch_size=300)

print(df.head())
print(f"Index stored at: {INDEX_DB}")
gpt_5_mini.print_total_usage()
