import os
from contextlib import asynccontextmanager

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import State
from starlette.middleware.cors import CORSMiddleware

from src.classification import ClassificationService
from src.preprocess import preprocess_fn
from src.util.type import SentimentRequest, Response
from src.controller.predict_controller import router as predict_controller

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler để load model khi ứng dụng khởi động."""
    print("Starting application...")
    app.state: State
    app.state.service = ClassificationService()
    yield  # Ứng dụng chạy tại đây
    print("Shutting down application...")


app = FastAPI(lifespan=lifespan)

# Thêm middleware
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["http://localhost:8182"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/predict-sentiment")
# # async def identify_face(request: SentimentRequest, service: ClassificationService = Depends(get_service)):
# async def predict_sentiment(comment: SentimentRequest, request: Request):
#     service: ClassificationService = request.app.state.service
#     content = preprocess_fn(comment.content.strip())
#
#     if not content:
#         raise HTTPException(status_code=400, detail="Content is required")
#
#     response = Response()
#     if len(content) == 1:
#         response.data = "Neutral"
#         return response.to_dict()
#
#     response.data = service.predict(content)
#     return response.to_dict()

app.include_router(predict_controller)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8111))  # Lấy port từ biến môi trường hoặc dùng 8000 mặc định
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
