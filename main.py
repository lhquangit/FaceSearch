from pathlib import Path

import cv2
import towhee
from pymilvus import Collection
from towhee._types import Image

import embedding
from connection import create_collection

DATASET = Path("data/rikai/")


def read_images(results):
    return [
        Image(cv2.imread(result.id), "BGR")
        for result in results
    ]


def main() -> None:

    collection = Collection("face_search")
    collection.load()

    (
        towhee
        .glob["path"]("test_images/*.png")
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "results"](collection=collection, limit=1)
        .runas_op["results", "result_images"](func=read_images)
        .select["image", "result_images"]()
        .show()
    )


if __name__ == "__main__":
    main()
