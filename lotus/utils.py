import base64
import os
import time
from io import BytesIO
from typing import Callable

import numpy as np
import pandas as pd
import requests  # type: ignore
from PIL import Image

import lotus


def cluster(col_name: str, ncentroids: int) -> Callable[[pd.DataFrame, int, bool], list[int]]:
    """
    Returns a function that clusters a DataFrame by a column using kmeans.

    Args:
        col_name (str): The column name to cluster by.
        ncentroids (int): The number of centroids to use.

    Returns:
        Callable: The function that clusters the DataFrame.
    """

    def ret(
        df: pd.DataFrame,
        niter: int = 20,
        verbose: bool = False,
        method: str = "kmeans",
    ) -> list[int]:
        import faiss

        """Cluster by column, and return a series in the dataframe with cluster-ids"""
        if col_name not in df.columns:
            raise ValueError(f"Column {col_name} not found in DataFrame")

        if ncentroids > len(df):
            raise ValueError(f"Number of centroids must be less than number of documents. {ncentroids} > {len(df)}")

        # get rmodel and index
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        try:
            col_index_dir = df.attrs["index_dirs"][col_name]
        except KeyError:
            raise ValueError(f"Index directory for column {col_name} not found in DataFrame")

        if vs.index_dir != col_index_dir:
            vs.load_index(col_index_dir)
        assert vs.index_dir == col_index_dir

        ids = df.index.tolist()  # assumes df index hasn't been resest and corresponds to faiss index ids
        vec_set = vs.get_vectors_from_index(col_index_dir, ids)
        d = vec_set.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(vec_set)

        # get nearest centroid to each vector
        scores, indices = kmeans.index.search(vec_set, 1)

        # get the cluster centroids
        # centroids = kmeans.centroids
        # return indices.flatten(), scores.flatten(), centroids
        return indices.flatten()

    return ret

def fetch_image(image, image_type: str = "Image"):
    """
    If `image` is a local file path (including 'file://' form), return it as a base64 data URL (PNG).
    Otherwise, return `image` unchanged.
    """
    if image is None:
        return None

    if isinstance(image, str):
        path = image[7:] if image.startswith("file://") else image
        if os.path.isfile(path):
            with Image.open(path) as img:
                buffered = BytesIO()
                img.convert("RGB").save(buffered, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return image


def show_safe_mode(estimated_cost, estimated_LM_calls):
    print(f"Estimated cost: {estimated_cost} tokens")
    print(f"Estimated LM calls: {estimated_LM_calls}")
    try:
        for i in range(5, 0, -1):
            print(f"Proceeding execution in {i} seconds... Press CTRL+C to cancel", end="\r")
            time.sleep(1)
            print(" " * 60, end="\r")
        print("\n")
    except KeyboardInterrupt:
        print("\nExecution cancelled by user")
        exit(0)
