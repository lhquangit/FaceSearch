from pathlib import Path

import towhee

import embedding
from connection import create_collection

DATASET = Path("data/rikai/")


def main() -> None:
    collection = create_collection("face_search", dim=512)

    dataset = (
        towhee
        .glob["path"](str(DATASET / "**/*.png"))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_insert.milvus[("path", "embedding"), "mr"]
    )

    for sample in dataset:
        print(sample.embedding.shape)


if __name__ == "__main__":
    main()
