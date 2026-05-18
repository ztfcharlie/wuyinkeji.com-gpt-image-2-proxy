"""测试 GPT-Image-2 同步代理服务，自动下载生成的图片"""

import requests
import base64
import sys
import time
import os
from datetime import datetime

BASE = "http://127.0.0.1:8000"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def save_b64_image(b64_data: str, index: int) -> str:
    """将 base64 图片数据保存到 output 目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{index}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return filepath


def download_image(url: str, index: int) -> str:
    """下载图片到 output 目录，返回本地文件路径"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{index}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(resp.content)

    return filepath


def test_openai_endpoint(prompt="一只在草地上奔跑的金毛犬，阳光明媚", response_format="url"):
    print(f"[OpenAI接口] prompt: {prompt}, response_format: {response_format}")
    start = time.time()

    resp = requests.post(f"{BASE}/v1/images/generations", json={
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": response_format,
    })
    elapsed = time.time() - start

    print(f"  状态码: {resp.status_code}")
    print(f"  耗时: {elapsed:.1f}s")

    data = resp.json()
    for i, img in enumerate(data.get("data", [])):
        if img.get("b64_json"):
            b64_data = img["b64_json"]
            path = save_b64_image(b64_data, i + 1)
            print(f"  图片{i+1}: b64_json ({len(b64_data)} chars)")
            print(f"  已保存: {path}")
        elif img.get("url"):
            url = img["url"]
            path = download_image(url, i + 1)
            print(f"  图片{i+1}: {url}")
            print(f"  已下载: {path}")

    print()


def test_simple_endpoint(prompt="a mountain landscape with snow"):
    print(f"[简洁接口] prompt: {prompt}")
    start = time.time()

    resp = requests.post(f"{BASE}/generate", json={
        "prompt": prompt,
        "size": "3:2",
    })
    elapsed = time.time() - start

    print(f"  状态码: {resp.status_code}")
    print(f"  耗时: {elapsed:.1f}s")

    data = resp.json()
    for i, url in enumerate(data.get("urls", [])):
        path = download_image(url, i + 1)
        print(f"  图片{i+1}: {url}")
        print(f"  已下载: {path}")

    print()


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    print("=" * 50)
    print(" GPT-Image-2 同步代理 - 测试脚本")
    print("=" * 50)
    print()

    if prompt:
        test_openai_endpoint(prompt)
    else:
        test_openai_endpoint(response_format="url")
        test_openai_endpoint(response_format="b64_json")
        test_simple_endpoint()

    print(f"图片保存在: {OUTPUT_DIR}")
