import uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
from classifier import ImageClassification

app = FastAPI(
    title="Image Classification",
    description="Classification API for images",
    version="1.0.0"
)

tags_metadata = [
    {
        "name": "classifiers",
        "description": "Classification endpoints, used to perform classification on images",
    }
]

image_classifier = ImageClassification()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class IndividualClass(BaseModel):
    Class: str = Field(..., title="Predicted Class")
    Pred_Prob: float = Field(..., title="Predicted Probability of image belonging to Class")


class ClassificationResponse(BaseModel):
    response: List[IndividualClass] = Field(..., title="List top class predictions")


@app.post("/image_classifier/", tags=["classifiers"], response_model=ClassificationResponse)
async def classify(input_file: UploadFile = File(...)):

    # Check uploaded file type
    if input_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid document type")

    # Try to open image file
    try:
        request_object_content = await input_file.read()
        img = Image.open(io.BytesIO(request_object_content))
    except Exception:
        return {"message": "There was an error uploading the file"}

    # Run inference
    output = image_classifier.classify(img)

    # Return response
    if type(output) == list:
        return {'response': output}
    else:
        raise TypeError("classifier output is not a list as expected")
