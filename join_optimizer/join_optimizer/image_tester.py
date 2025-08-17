import duckdb
import os
from lotus.dtype_extensions import ImageArray
import pandas as pd

from lotus.sem_ops import sem_captions_index

parquet_path_sampeled = 'fashion_product_images/styles.parquet'
parquet_path = 'fashion_product_images/styles.parquet'
details_path = 'fashion_product_images/styles_details.parquet'
sample_size = 0.0005
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

""").to_df()


image_file_names = df["id"]
image_URLs = df["imageURL"]
image_paths = [os.path.join("fashion_product_images/images_resized", str(image) + ".jpg") for image in image_file_names]
df2 = pd.DataFrame({"image": ImageArray(image_URLs), "label": image_file_names, "image_path": image_paths, "image_URLs": image_URLs , "articleType": df["articleType"], "baseColour": df["baseColour"]})
merged_df = pd.merge(df, df2,  left_on='id', right_on='label')
merged_df.columns = ['_' + col for col in merged_df.columns]
from lotus.vector_store import FaissVS
import lotus
from lotus.models import LM, SentenceTransformersRM

gpt_4o_mini = LM("gpt-4o-mini")
gpt_4o = LM("gpt-4o-mini")

# CLIP embedding model – works for both text & image
# rm  = SentenceTransformersRM(model="clip-ViT-B-32")
# rm  = SentenceTransformersRM(model="clip-ViT-L-14", max_batch_size=32)

lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)
merged_df= df = merged_df.sem_captions_index(
    img_col="_image",
    out_col="caption",
    db_path="caps.db",
    batch_size=32,
    overwrite_db=True,
)
print(merged_df[["_image", "caption"]].head(10))
