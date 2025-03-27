import os
from contextlib import asynccontextmanager

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware

from src.Constant import CLASSIFICATION_PATH
from src.classification import ClassificationService
from src.preprocess import preprocess_fn
from src.type import SentimentRequest, Response

# Khởi tạo service quản lý model
classification_service = ClassificationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler để load model khi ứng dụng khởi động."""
    print("Starting application...")
    classification_service.load_components(CLASSIFICATION_PATH)
    yield  # Ứng dụng chạy tại đây
    print("Shutting down application...")


app = FastAPI(lifespan=lifespan)


# Dependency để inject service vào controller
def get_service() -> ClassificationService:
    return classification_service


# Thêm middleware nếu cần (ví dụ: CORS)
app.add_middleware(
    CORSMiddleware,# type: ignore
    allow_origins=["http://localhost:8182"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def root1():
    return {"data": "Hello World"}

@app.get("/")
async def root2():
    return {"data": "Hello World"}

@app.post("/predict-sentiment")
async def identify_face(request: SentimentRequest, service: ClassificationService = Depends(get_service)):
    content = preprocess_fn(request.content.strip())

    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    response = Response()
    if len(content) == 1:
        response.data = "Neutral"
        return response.to_dict()

    response.data = service.predict(content)
    return response.to_dict()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8111))  # Lấy port từ biến môi trường hoặc dùng 8000 mặc định
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
