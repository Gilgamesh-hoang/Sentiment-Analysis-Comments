from fastapi import FastAPI, Request, HTTPException
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from decouple import config
import base64

app = FastAPI()

PUBLIC_KEY_BASE64 = config("PUBLIC_KEY_BASE64")
PUBLIC_KEY = base64.b64decode(PUBLIC_KEY_BASE64).decode("utf-8")
ALGORITHM = "RS256" 

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Bỏ qua kiểm tra cho một số endpoint nếu cần (ví dụ: /docs, /openapi.json)
        if request.url.path in ["/docs", "/openapi.json"]:
            return await call_next(request)

        # Lấy token từ header Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=403, detail="Authorization header missing or invalid")

        token = auth_header.split("Bearer ")[1]
        try:
            # Xác thực JWT chỉ với public key
            jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM])
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=403, detail="Invalid or expired token")
        except jwt.InvalidSignatureError:
            raise HTTPException(status_code=403, detail="Invalid token signature")

        # Nếu token hợp lệ, tiếp tục xử lý request
        response = await call_next(request)
        return response

# Thêm middleware vào app
app.add_middleware(JWTAuthMiddleware)