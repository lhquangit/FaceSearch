from pathlib import Path

import cv2
import towhee
from towhee._types import Image

import embedding
from connection import create_collection

DATASET = Path("data/rikai/")
TEST_IMAGE = Path("test_image.png")


def read_images(results):
    return [
        Image(cv2.imread(result.id), "BGR")
        for result in results
    ]


def main() -> None:
    collection = create_collection("face_search", dim=512)

    (
        towhee
        .glob["path"](str(DATASET / "**/*.png"))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_insert.milvus[("path", "embedding"), "mr"](collection=collection)
    )

    collection.load()

    (
        towhee
        .glob["path"](str(TEST_IMAGE))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "results"](collection=collection, limit=1)
        .runas_op["results", "result_images"](func=read_images)
        .select["image", "result_images"]()
        .show()
    )


if __name__ == "__main__":
    main()
