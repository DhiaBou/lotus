import os
from dotenv import load_dotenv
import kagglehub
import duckdb
import pyarrow.parquet as pq


def download_fashion_product_images():
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    print(f"Download complete. Dataset available at: {dataset_path}")
    return dataset_path


def postprocess_data(kaggle_cache_path):
    out_dir = 'fashion_product_images'
    os.makedirs(out_dir, exist_ok=True)

    # Process styles.csv
    output_path = os.path.join(out_dir, 'styles.parquet')
    print(f"Creating {output_path}")
    if not os.path.exists(output_path):
        duckdb.sql(f"""
            COPY (
                SELECT *
                FROM read_csv('{os.path.join(kaggle_cache_path, 'fashion-dataset', 'styles.csv')}', header=true, delim=',', quote='"', ignore_errors=true)
            )
            TO '{output_path}' (FORMAT PARQUET)
        """)

    # Process images.csv
    output_path = os.path.join(out_dir, 'image_mapping.parquet')
    print(f"Creating {output_path}")
    if not os.path.exists(output_path):
        duckdb.sql(f"""
            COPY (
                SELECT split_part(filename, '.', 1) as id, filename, link
                FROM read_csv('{os.path.join(kaggle_cache_path, 'fashion-dataset', 'images.csv')}', header=true)
            )
            TO '{output_path}' (FORMAT PARQUET)
        """)

    # Process styles/*.json files
    output_path = os.path.join(out_dir, 'styles_details.parquet')
    print(f"Creating {output_path}")
    if not os.path.exists(output_path):
        with duckdb.connect() as con:
            con.execute("SET threads = 2;")
            con.execute("SET preserve_insertion_order=false;")
            con.execute(f"""
            COPY (
                SELECT unnest(data) 
                FROM read_json('{os.path.join(kaggle_cache_path, 'fashion-dataset', 'styles', '*.json')}', union_by_name=true)
            )
            TO '{output_path}' (FORMAT PARQUET)
            """)
        print(f"Parquet file created at: {output_path}")

    # Process images/*.jpg files
    output_path = os.path.join(out_dir, 'images')
    print(f"Linking images to {output_path}")
    if not os.path.exists(output_path):
        os.symlink(os.path.join(kaggle_cache_path, 'fashion-dataset', 'images'), output_path)


def create_sample(sample_size=0.1):
    print(f"Creating sample with size: {sample_size * 100}%")
    input_dir = 'fashion_product_images'
    out_dir = f'fashion_product_images_{str(sample_size).replace(".", "")}'
    os.makedirs(out_dir, exist_ok=True)

    # Create a sample from one table; the others can then be joined to it.
    duckdb.sql(f"""
        COPY (
            SELECT *
            FROM read_parquet('{os.path.join(input_dir, 'styles.parquet')}')
            USING SAMPLE {sample_size * 100} PERCENT (reservoir, 42)
        )
        TO '{os.path.join(out_dir, 'styles.parquet')}' (FORMAT PARQUET)
    """)

    duckdb.sql(f"""
        COPY (
            SELECT styles_details.*
            FROM read_parquet('{os.path.join(out_dir, 'styles.parquet')}') AS styles
            JOIN read_parquet('{os.path.join(input_dir, 'styles_details.parquet')}') AS styles_details
            ON styles_details.id = styles.id
        )
        TO '{os.path.join(out_dir, 'styles_details.parquet')}' (FORMAT PARQUET)
    """)

    duckdb.sql(f"""
        COPY (
            SELECT image_mapping.*
            FROM read_parquet('{os.path.join(out_dir, 'styles.parquet')}') AS styles
            JOIN read_parquet('{os.path.join(input_dir, 'image_mapping.parquet')}') AS image_mapping
            ON image_mapping.id = styles.id
        )
        TO '{os.path.join(out_dir, 'image_mapping.parquet')}' (FORMAT PARQUET)
    """)

    # Symlink necessary image files
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    table = pq.read_table(os.path.join(out_dir, 'image_mapping.parquet'))
    for filename in table['filename']:
        src = os.path.join(input_dir, 'images', filename.as_py())
        dst = os.path.join(out_dir, 'images', filename.as_py())
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(src), dst)


if __name__ == "__main__":
    load_dotenv()
    kaggle_cache_path = download_fashion_product_images()
    postprocess_data(kaggle_cache_path)
    create_sample(sample_size=0.01)
