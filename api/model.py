
from io import BytesIO

import cv2
import towhee
from PIL import Image

from fastapi import FastAPI, File
from pymilvus import Collection, connections

from ..embedding import predict_face


def read_images(results):
    from towhee._types import Image

    return [
        Image(cv2.imread(result.id), "BGR")
        for result in results
    ]


app = FastAPI()
connections.connect(host="127.0.0.1", port="19530")
collection = Collection("face_search")
collection.load()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World!"}


@app.post("/similar_images")
async def similar_images(image: bytes = File()) -> dict[str, str]:
    pil_image = Image.open(BytesIO(image))
    similar_images = (
        towhee
        .glob["path"]("../test_images/*.png")
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "results"](collection=collection, limit=1)
        .runas_op["results", "result_images"](func=read_images)
        .select["image", "result_images"]()
    )
    print(similar_images)
    return {"similar_images": "Hello, World!"}
