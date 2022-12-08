from io import BytesIO
from pathlib import Path

import cv2
import towhee
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


from fastapi import FastAPI, File
from fastapi.responses import FileResponse
from pymilvus import Collection, connections

from model import predict_face

TEST_DIR = Path("test_images/")
N_SIMILAR_IMAGES = 1
SAME_PERSON_THRESHOLD = 0.2


def read_paths(results):
    return [result.id for result in results]


app = FastAPI()
connections.connect(host="127.0.0.1", port="19530")
collection = Collection("face_search")
collection.load()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World!"}


@app.post("/similar_images")
async def similar_images(image: bytes = File()) -> FileResponse:
    pil_image = Image.open(BytesIO(image))
    pil_image.save(str(TEST_DIR / "test_image.png"))
    results = (
        towhee
        .glob["path"](str(TEST_DIR / "test_image.png"))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "results"](collection=collection, limit=N_SIMILAR_IMAGES)
        .runas_op["results", "result_paths"](func=read_paths)
        .select["image", "result_paths"]()
    )
    result_paths = [results[0].result_paths[idx] for idx in range(N_SIMILAR_IMAGES)]
    return [FileResponse(result_path) for result_path in result_paths][0]


@app.post("/same_person")
async def same_person(image_1: bytes = File(), image_2: bytes = File()) -> bool:
    pil_image_1 = Image.open(BytesIO(image_1))
    pil_image_1.save(str(TEST_DIR / "test_image_1.png"))
    pil_image_2 = Image.open(BytesIO(image_2))
    pil_image_2.save(TEST_DIR / "test_image_2.png")
    embeddings = (
        towhee
        .glob["path"](TEST_DIR / "test_image_*.png")
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
    )
    return bool(
        cosine_similarity(
            embeddings[0].embedding.reshape(1, -1), 
            embeddings[1].embedding.reshape(1, -1)
        ).squeeze() > SAME_PERSON_THRESHOLD
    )
