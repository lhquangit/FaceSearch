from io import BytesIO

import numpy as np
import towhee
from PIL import Image
from fastapi import FastAPI, File
from fastapi.responses import FileResponse
from pymilvus import connections
from sklearn.metrics.pairwise import cosine_similarity

from database import create_collection
from model import predict_face

DATASET = "dataset/rikai"
TEST_DIR = "test_images"
SAME_PERSON_THRESHOLD = 0.2


# Establish the database connection
connections.connect(host="127.0.0.1", port="19530")
collection = create_collection("face_search", dim=512)
collection.load()

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World!"}


@app.post("/similar_image")
async def similar_image(image: bytes = File()) -> FileResponse:
    pil_image = Image.open(BytesIO(image))
    pil_image.save(f"{TEST_DIR}/test_image.png")

    data_collection = (
        towhee.glob["path"](f"{TEST_DIR}/test_image.png")
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "result"](collection=collection, limit=1)
        .runas_op["result", "result_path"](
            func=lambda results: [result.id for result in results]
        )
        .select["result_path"]()
    )

    sample = data_collection[0]
    result_path = sample.result_path[0]
    return FileResponse(result_path, media_type="image/png")


@app.post("/same_person")
async def same_person(
    image_1: bytes = File(media_type="image/png"),
    image_2: bytes = File(media_type="image/png"),
) -> bool:
    images = [image_1, image_2]
    pil_images = [Image.open(BytesIO(image)) for image in images]
    for idx, pil_image in enumerate(pil_images, start=1):
        pil_image.save(f"{TEST_DIR}/test_image_{idx}.png")

    data_collection = (
        towhee.glob["path"](f"{TEST_DIR}/test_image_*.png")
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
    )

    embeddings = [sample.embedding for sample in data_collection]
    embeddings = [
        np.expand_dims(embedding, axis=0) for embedding in embeddings
    ]  # Expand the dimension of embedding for scikit-learn API compatability
    similarity = cosine_similarity(*[embedding for embedding in embeddings]).squeeze()
    return bool(similarity > SAME_PERSON_THRESHOLD)


@app.post("/add_user")
async def add_user(path: str, image: bytes = File(media_type="image/png")) -> None:
    pil_image = Image.open(BytesIO(image))
    pil_image.save(f"{DATASET}/{path}.png")
    embedding = predict_face(np.asarray(pil_image))
    sample = [[path], [embedding]]
    collection.insert(sample)
