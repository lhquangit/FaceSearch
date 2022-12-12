from pathlib import Path

from pymilvus import connections
import towhee

from database import create_collection
from model import predict_face

DATASET = Path("dataset/rikai/")


def main():
    connections.connect(host="127.0.0.1", port="19530")
    collection = create_collection("face_search", dim=512)

    (
        towhee
        .glob["path"](str(DATASET / "**/*.png"))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_insert.milvus[("path", "embedding"), "mr"](collection=collection)
    )


if __name__ == "__main__":
    main()
