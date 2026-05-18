"""
GPT-Image-2 高并发同步代理服务

将 wuyinkeji 异步 API 包装为兼容 OpenAI /v1/images/generations 的同步接口。

启动（单 worker）:
  uvicorn server:app --host 0.0.0.0 --port 8000

启动（多 worker，推荐生产环境）:
  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import base64
import json
import os
import time
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy")

# ── 上游 API 配置 ──────────────────────────────────────────────
UPSTREAM_SUBMIT_URL = "https://api.wuyinkeji.com/api/async/image_gpt"
UPSTREAM_POLL_URL = "https://api.wuyinkeji.com/api/async/detail"
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "lA0g3L9wSnM9T70Y6hsxnspZDm")
POLL_INTERVAL = 3
MAX_POLL_TIME = 300

# ── 认证配置 ───────────────────────────────────────────────────
API_TOKEN = os.getenv("API_TOKEN", "")

# ── 并发配置 ───────────────────────────────────────────────────
MAX_CONCURRENT_SUBMITS = 200
MAX_CONCURRENT_DOWNLOADS = 500
HTTPX_MAX_CONNECTIONS = 2000
HTTPX_MAX_KEEPALIVE = 500

# ── 日志配置 ───────────────────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

# ── 输出与清理配置 ─────────────────────────────────────────────
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
IMAGE_TTL_SECONDS = int(os.getenv("IMAGE_TTL_SECONDS", "900"))  # 默认 15 分钟

# ── 全局资源 ───────────────────────────────────────────────────
submit_semaphore: asyncio.Semaphore
download_semaphore: asyncio.Semaphore
http_client: httpx.AsyncClient
cleanup_task: asyncio.Task


# ── 日志记录 ──────────────────────────────────────────────────
def write_log(record: dict):
    """将请求日志写入按日分界的日志文件"""
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{date_str}.log")
    line = json.dumps(record, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── OpenAI 标准错误响应 ──────────────────────────────────────
def openai_error_response(
    status_code: int,
    message: str,
    error_type: str = "server_error",
    error_code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    """生成符合 OpenAI 格式的错误响应"""
    body = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": error_code or error_type,
        }
    }
    return JSONResponse(status_code=status_code, content=body)


# ── 自定义异常（携带 OpenAI 错误信息） ──────────────────────
class UpstreamError(Exception):
    def __init__(self, status_code: int, message: str, error_type: str, error_code: str | None = None):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.error_code = error_code or error_type


# ── 生命周期 ──────────────────────────────────────────────────
async def _cleanup_old_images():
    """后台循环：删除超过 IMAGE_TTL_SECONDS 的图片文件"""
    while True:
        await asyncio.sleep(60)
        try:
            if not os.path.isdir(OUTPUT_DIR):
                continue
            cutoff = time.time() - IMAGE_TTL_SECONDS
            count = 0
            for fname in os.listdir(OUTPUT_DIR):
                fpath = os.path.join(OUTPUT_DIR, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    count += 1
            if count:
                logger.info("已清理 %d 张过期图片", count)
        except Exception:
            logger.exception("清理图片失败")


@asynccontextmanager
async def lifespan(application: FastAPI):
    global submit_semaphore, download_semaphore, http_client, cleanup_task
    submit_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUBMITS)
    download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=HTTPX_MAX_CONNECTIONS,
            max_keepalive_connections=HTTPX_MAX_KEEPALIVE,
        ),
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    cleanup_task = asyncio.create_task(_cleanup_old_images())
    logger.info("服务启动 | 最大提交并发=%d | 最大下载并发=%d | 连接池=%d | 图片保留=%ds",
                MAX_CONCURRENT_SUBMITS, MAX_CONCURRENT_DOWNLOADS, HTTPX_MAX_CONNECTIONS, IMAGE_TTL_SECONDS)
    yield
    cleanup_task.cancel()
    await http_client.aclose()
    logger.info("服务关闭")


app = FastAPI(title="GPT-Image-2 Sync Proxy", lifespan=lifespan)


# ── Bearer Token 认证中间件 ────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_TOKEN:
        if request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != API_TOKEN:
            return openai_error_response(
                status_code=401,
                message="Incorrect API key provided",
                error_type="invalid_request_error",
                error_code="invalid_api_key",
            )
    return await call_next(request)


# ── 全局异常处理 → OpenAI 格式 ──────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = "; ".join(
        f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in errors
    )
    return openai_error_response(
        status_code=422,
        message=messages,
        error_type="invalid_request_error",
        error_code="invalid_request_error",
    )


@app.exception_handler(UpstreamError)
async def upstream_error_handler(request: Request, exc: UpstreamError):
    return openai_error_response(exc.status_code, exc.message, exc.error_type, exc.error_code)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "type" in exc.detail:
        return openai_error_response(
            exc.status_code,
            exc.detail.get("message", str(exc.detail)),
            exc.detail.get("type", "server_error"),
        )
    return openai_error_response(exc.status_code, str(exc.detail), "server_error")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常")
    return openai_error_response(500, "Internal server error", "server_error", "internal_error")


# ── OpenAI 兼容请求模型 ────────────────────────────────────────
class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-image-2"
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = "auto"
    quality: Optional[str] = "auto"
    response_format: Optional[str] = Field(default="b64_json", pattern="^(url|b64_json)$")
    output_format: Optional[str] = Field(default="png", pattern="^(png|jpeg|webp)$")
    output_compression: Optional[int] = Field(default=100, ge=0, le=100)
    user: Optional[str] = None
    reference_images: Optional[list[str]] = Field(default=None, description="参考图片URL列表")

    def to_upstream_size(self) -> str:
        mapping = {
            "1024x1024": "1:1",
            "1536x1024": "3:2",
            "1024x1536": "2:3",
            "2048x2048": "1:1",
            "auto": "auto",
        }
        return mapping.get(self.size, "auto")


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class UsageInfo(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ImageResponse(BaseModel):
    created: int
    data: list[ImageData]
    usage: Optional[UsageInfo] = None


# ── 核心逻辑 ──────────────────────────────────────────────────
async def submit_and_wait(
    prompt: str, size: str, reference_urls: list[str] | None = None
) -> tuple[list[str], str]:
    """提交生成请求并等待结果，返回 (图片URL列表, 上游任务ID)"""
    async with submit_semaphore:
        payload: dict = {"prompt": prompt, "size": size}
        if reference_urls:
            payload["urls"] = reference_urls
        resp = await http_client.post(
            UPSTREAM_SUBMIT_URL,
            headers={"Authorization": UPSTREAM_KEY, "Content-Type": "application/json"},
            json=payload,
        )
        body = resp.json()
        if body.get("code") != 200:
            raise UpstreamError(
                status_code=502,
                message=f"Upstream submit failed: {body.get('msg')}",
                error_type="server_error",
                error_code="upstream_error",
            )

        task_id = body["data"]["id"]

    start = time.time()
    while time.time() - start < MAX_POLL_TIME:
        await asyncio.sleep(POLL_INTERVAL)

        resp = await http_client.get(UPSTREAM_POLL_URL, params={"key": UPSTREAM_KEY, "id": task_id})
        body = resp.json()
        task_data = body.get("data", {})
        status = task_data.get("status")

        if status == 2:
            return task_data.get("result") or [], task_id
        if status in (-1, 3):
            msg = task_data.get("message", "未知错误")
            raise UpstreamError(
                status_code=502,
                message=f"Image generation failed: {msg}",
                error_type="server_error",
                error_code="generation_failed",
            )

    raise UpstreamError(
        status_code=504,
        message="Image generation timed out",
        error_type="server_error",
        error_code="timeout",
    )


async def download_as_b64(url: str) -> str:
    async with download_semaphore:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")


# ── OpenAI 兼容端点 ───────────────────────────────────────────
@app.post("/v1/images/generations")
async def openai_compatible(req: ImageRequest, request: Request):
    request_id = f"img_{uuid.uuid4().hex[:16]}"
    start_time = time.time()
    log_record: dict = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/v1/images/generations",
        "prompt": req.prompt[:200],
        "size": req.size,
        "response_format": req.response_format,
        "reference_images": req.reference_images,
        "user": req.user,
    }

    try:
        upstream_size = req.to_upstream_size()
        urls, upstream_task_id = await submit_and_wait(req.prompt, upstream_size, req.reference_images)

        images = []
        if req.response_format == "b64_json":
            tasks = [download_as_b64(url) for url in urls]
            b64_results = await asyncio.gather(*tasks)
            for b64 in b64_results:
                images.append(ImageData(b64_json=b64, revised_prompt=req.prompt))
        else:
            for url in urls:
                images.append(ImageData(url=url, revised_prompt=req.prompt))

        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "success",
            "upstream_task_id": upstream_task_id,
            "image_count": len(urls),
            "result_urls": urls,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)

        return ImageResponse(
            created=int(time.time()),
            data=images,
            usage=UsageInfo(total_tokens=0, input_tokens=0, output_tokens=0),
        )

    except UpstreamError as e:
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "failed",
            "error_type": e.error_type,
            "error_code": e.error_code,
            "error_message": e.message,
            "http_status": e.status_code,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        raise

    except httpx.HTTPStatusError as e:
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "failed",
            "error_type": "server_error",
            "error_code": "download_failed",
            "error_message": f"Failed to download image: HTTP {e.response.status_code}",
            "http_status": 502,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        raise UpstreamError(502, f"Failed to download generated image: HTTP {e.response.status_code}", "server_error", "download_failed")

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "failed",
            "error_type": "server_error",
            "error_code": "internal_error",
            "error_message": str(e),
            "http_status": 500,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        raise


# ── 简洁接口 ──────────────────────────────────────────────────
class SimpleRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1:1"
    urls: Optional[list[str]] = Field(default=None, description="参考图片URL列表")


@app.post("/generate")
async def simple_generate(req: SimpleRequest):
    request_id = f"img_{uuid.uuid4().hex[:16]}"
    start_time = time.time()
    log_record: dict = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/generate",
        "prompt": req.prompt[:200],
        "size": req.size,
        "reference_urls": req.urls,
    }

    try:
        urls, upstream_task_id = await submit_and_wait(req.prompt, req.size, req.urls)
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "success",
            "upstream_task_id": upstream_task_id,
            "image_count": len(urls),
            "result_urls": urls,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        return {"urls": urls}

    except UpstreamError as e:
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "failed",
            "error_type": e.error_type,
            "error_code": e.error_code,
            "error_message": e.message,
            "http_status": e.status_code,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        raise

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        log_record.update({
            "status": "failed",
            "error_type": "server_error",
            "error_code": "internal_error",
            "error_message": str(e),
            "http_status": 500,
            "elapsed_seconds": elapsed,
        })
        write_log(log_record)
        raise


# ── 健康检查 ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "submit_semaphore": f"{MAX_CONCURRENT_SUBMITS - submit_semaphore._value}/{MAX_CONCURRENT_SUBMITS}",
        "download_semaphore": f"{MAX_CONCURRENT_DOWNLOADS - download_semaphore._value}/{MAX_CONCURRENT_DOWNLOADS}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
