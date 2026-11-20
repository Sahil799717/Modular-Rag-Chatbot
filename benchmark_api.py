
import math
import time
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

import httpx

from app.core.security import create_access_token
from app.infrastructure.cache_service import cache_delete
from app.rag.retrieval_cache import (
    get_retrieval_cache_key,
    get_user_corpus_version,
)


BASE_URL = "http://127.0.0.1:8000"
FILE_PATH = Path("Temple.pdf")

MISS_RUNS = 2
HIT_RUNS = 3

QUESTION = (
    "What is the difference between a shikhara "
    "and a vimana in Indian temple architecture?"
)


def report(title, values):
    if not values:
        print(f"\n{title}: No successful requests")
        return

    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]

    print(f"\n{title}")
    print(f"Requests: {len(values)}")
    print(f"Mean:     {mean(values):.2f} ms")
    print(f"Median:   {median(values):.2f} ms")
    print(f"P95:      {p95:.2f} ms")
    print(f"Min:      {min(values):.2f} ms")
    print(f"Max:      {max(values):.2f} ms")
    print(f"RPS:      {1000 / mean(values):.2f}")


def benchmark():
    if not FILE_PATH.exists():
        raise FileNotFoundError(FILE_PATH)

    user_id = f"api-benchmark-{uuid4().hex}"
    token = create_access_token(user_id)

    headers = {"Authorization": f"Bearer {token}"}

    document_id = None
    cache_key = None
    uploaded_file = None

    miss_times = []
    hit_times = []
    errors = 0

    with httpx.Client(
        base_url=BASE_URL,
        headers=headers,
        timeout=90.0
    ) as client:

        try:
            print("Uploading benchmark document...")

            with FILE_PATH.open("rb") as file:
                response = client.post(
                    "/documents/upload",
                    files={
                        "file": (
                            "Temple.pdf",
                            file,
                            "application/pdf"
                        )
                    }
                )

            response.raise_for_status()

            result = response.json()
            document_id = result["document_id"]

            uploaded_file = Path("uploads") / f"{document_id}.pdf"

            print("Document uploaded:", document_id)

            version = get_user_corpus_version(user_id)

            cache_key = get_retrieval_cache_key(
                question=QUESTION,
                user_id=user_id,
                document_id=document_id,
                corpus_version=version,
            )

            def send_chat():
                start = time.perf_counter()

                response = client.post(
                    "/chat",
                    json={
                        "question": QUESTION,
                        "document_id": document_id
                    }
                )

                elapsed = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text}"
                    )

                return elapsed

            print("\nRunning cache-miss benchmark...")

            for i in range(MISS_RUNS):
                cache_delete(cache_key)

                try:
                    elapsed = send_chat()
                    miss_times.append(elapsed)
                    print(f"Miss {i + 1}: {elapsed:.2f} ms")

                except Exception as e:
                    errors += 1
                    print(f"Miss {i + 1} failed: {e}")

            print("\nRunning cache-hit benchmark...")

            for i in range(HIT_RUNS):
                try:
                    elapsed = send_chat()
                    hit_times.append(elapsed)
                    print(f"Hit {i + 1}: {elapsed:.2f} ms")

                except Exception as e:
                    errors += 1
                    print(f"Hit {i + 1} failed: {e}")

            report("API CACHE MISS RESULTS", miss_times)
            report("API CACHE HIT RESULTS", hit_times)

            total = len(miss_times) + len(hit_times)

            print("\nAPI BENCHMARK SUMMARY")
            print(f"Successful requests: {total}")
            print(f"Failed requests:     {errors}")

            if total:
                print(
                    f"Error rate:          "
                    f"{errors / (total + errors) * 100:.2f}%"
                )

            if miss_times and hit_times:
                speedup = mean(miss_times) / mean(hit_times)

                print(f"API latency speedup: {speedup:.2f}x")

        finally:
            if document_id:
                try:
                    response = client.delete(
                        f"/documents/{document_id}"
                    )
                    print("\nDocument cleanup:", response.status_code)
                except Exception as e:
                    print("Document cleanup failed:", e)

            if cache_key:
                try:
                    cache_delete(cache_key)
                except Exception:
                    pass

            if uploaded_file and uploaded_file.exists():
                uploaded_file.unlink()
                print("Temporary PDF removed")


if __name__ == "__main__":
    benchmark()
