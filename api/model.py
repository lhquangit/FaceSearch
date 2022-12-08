from fastapi import FastAPI


app = FastAPI()


@app.get("/similar_images/{base64_image}")
async def similar_images(base64_image: str) -> dict[str, list[str]]:
    return {"similar_images": [base64_image]}
