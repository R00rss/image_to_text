from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from os import path
import functions.IA as ml_functions


origins = ["*"]

API_KEY = "mysecretapikeyIA"

project_path = path.dirname(path.realpath(__file__))
uploads_path = os.path.join(project_path, "uploads")


async def validate_token_query(api_key: str = Query()):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate_text_by_image")
async def generate_text_by_image(
    api_key: None = Depends(validate_token_query),
    image: UploadFile = File(...),
):
    try:
        if not os.path.exists(uploads_path):
            os.makedirs(uploads_path)

        file_location = os.path.join(uploads_path, image.filename)
        with open(file_location, "wb") as file:
            file.write(image.file.read())
        res = ml_functions.my_predict(file_location)
        print(res)
        return {"text": res}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_connection")
async def test(api_key: None = Depends(validate_token_query)):
    return {"hello": "world"}


@app.get("/test_connection2")
async def test2():
    return {"hello": "world"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=True,
        port=2001,
    )
