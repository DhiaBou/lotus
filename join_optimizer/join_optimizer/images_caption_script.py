import os

import duckdb
import pandas as pd

from lotus.dtype_extensions import ImageArray
from lotus.models.blip_cm import BlipCaptioner
from lotus.fts_store.db_fts_store import SQLiteFTSStore

parquet_path_sampeled = "fashion_product_images/styles.parquet"
parquet_path = "fashion_product_images/styles.parquet"
details_path = "fashion_product_images/styles_details.parquet"
sample_size = 0.02
df = duckdb.query(f"""
with images as (
    SELECT *
    FROM parquet_scan('{parquet_path}')
    USING SAMPLE {sample_size * 100} PERCENT (reservoir, 80)
    )
    select
     images.id ,images.subcategory, images.articletype, images.basecolour, details.price, images.productDisplayName, styleimages.default.resolutions."360X480"  as imageURL
    -- *
    from images, parquet_scan('{details_path}') details
    where images.id = details.id
    -- and details.price <1000
    order by images.id

""").to_df()


image_file_names = df["id"]
image_URLs = df["imageURL"]
image_paths = [os.path.join("fashion_product_images/images", str(image) + ".jpg") for image in image_file_names]
df2 = pd.DataFrame(
    {
        "image": ImageArray(image_paths),
        "label": image_file_names,
        "image_path": image_paths,
        "image_URLs": image_URLs,
        "articleType": df["articleType"],
        "baseColour": df["baseColour"],
    }
)
merged_df = pd.merge(df, df2, left_on="id", right_on="label")
merged_df.columns = ["_" + col for col in merged_df.columns]
import lotus
from lotus.models import LM

gpt_4o_mini = LM("gpt-4o-mini")
gpt_4o = LM("gpt-4o-mini")

# CLIP embedding model – works for both text & image
# rm  = SentenceTransformersRM(model="clip-ViT-B-32")
# rm  = SentenceTransformersRM(model="clip-ViT-L-14", max_batch_size=32)
db_path = "caps_blip.db"
lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini, cm=BlipCaptioner(), cs=SQLiteFTSStore())

# merged_df = merged_df.sem_captions_index("_image", index_dir=db_path)
merged_df = merged_df.sem_captions_index.attach_index("_image", index_dir=db_path)
merged_df = merged_df.sem_captions_index.load("_image")
merged_df = merged_df.sem_captions_index.search("white man shirt", "_image", K=120)

print(merged_df)
