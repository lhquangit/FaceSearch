from io import BytesIO
from pathlib import Path

import numpy as np
import towhee
from PIL import Image
from fastapi import FastAPI, File
from fastapi.responses import FileResponse
from pymilvus import connections
from sklearn.metrics.pairwise import cosine_similarity

from database import create_collection
from model import predict_face

TEST_DIR = Path("test_images/")
SAME_PERSON_THRESHOLD = 0.5


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
    pil_image.save(str(TEST_DIR / "test_image.png"))

    data_collection = (
        towhee
        .glob["path"](str(TEST_DIR / "test_image.png"))
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
        .ann_search.milvus["embedding", "result"](collection=collection)
        .runas_op["result", "result_path"](
            func=lambda results: [result.id for result in results]
        )
        .select["result_path"]()
    )

    sample = data_collection[0]  # Grab the only sample in the data collection
    result_path = sample.result_path
    print(result_path)
    return FileResponse(result_path)

#
# @app.post("/same_person")
# async def same_person(images: tuple[bytes] = (File(), File())) -> bool:
#     pil_images = [
#         Image.open(BytesIO(image)) for image in images
#     ]
#     for idx, pil_image in enumerate(pil_images, start=1):
#         pil_image.save(TEST_DIR / f"test_image_{idx}.png")
#
#     data_collection = (
#         towhee
#         .glob["path"](TEST_DIR / "test_image_*.png")
#         .image_decode["path", "image"]()
#         .extract_embedding["image", "embedding"]()
#     )
#
#     embeddings = [sample.embedding for sample in data_collection]
#     embeddings = [
#         np.expand_dims(embedding, axis=0) for embedding in embeddings
#     ]  # Expand the dimension of embedding for scikit-learn API compatability
#     similarity = cosine_similarity(*[embedding for embedding in embeddings]).squeeze()
#     return bool(similarity > SAME_PERSON_THRESHOLD)
