from fastapi import APIRouter
from fastapi import HTTPException, Request

from src.classification import ClassificationService
from src.preprocess import preprocess_fn
from src.util.type import Response
from src.util.type import SentimentRequest

router = APIRouter()

@router.post("/predict-sentiment")
async def predict_sentiment(comment: SentimentRequest, request: Request):
    service: ClassificationService = request.app.state.service
    content = preprocess_fn(comment.content.strip())

    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    response = Response()
    if len(content) == 1:
        response.data = "Neutral"
        return response.to_dict()

    response.data = service.predict(content)
    return response.to_dict()