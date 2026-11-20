
import time
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

from app.rag.ingestion import ingest_pdf
from app.rag.retrieval_cache import (
    get_cached_documents,
    get_retrieval_cache_key,
    get_user_corpus_version,
)
from app.infrastructure.cache_service import cache_delete
from app.services.vector_service import delete_document


FILE_PATH = Path(__file__).parent / "Temple.pdf"

USER_ID = f"benchmark-user-{uuid4().hex}"
DOCUMENT_ID = f"benchmark-doc-{uuid4().hex}"

QUESTION = (
    "What is the difference between a shikhara "
    "and a vimana in Indian temple architecture?"
)

MISS_RUNS = 5
HIT_RUNS = 30


def report(title, values):
    print(f"\n{title}")
    print(f"Requests: {len(values)}")
    print(f"Mean:     {mean(values):.2f} ms")
    print(f"Median:   {median(values):.2f} ms")
    print(f"Min:      {min(values):.2f} ms")
    print(f"Max:      {max(values):.2f} ms")


def retrieve():
    start = time.perf_counter()

    documents = get_cached_documents(
        question=QUESTION,
        user_id=USER_ID,
        document_id=DOCUMENT_ID,
    )

    elapsed = (time.perf_counter() - start) * 1000

    assert documents, "No documents retrieved"

    return elapsed


def benchmark():
    if not FILE_PATH.exists():
        raise FileNotFoundError(FILE_PATH)

    cache_key = None

    try:
        print("Ingesting benchmark document...")

        ingest_pdf(
            file_path=str(FILE_PATH),
            user_id=USER_ID,
            document_id=DOCUMENT_ID,
            original_file_name="Temple.pdf",
        )

        version = get_user_corpus_version(USER_ID)

        cache_key = get_retrieval_cache_key(
            question=QUESTION,
            user_id=USER_ID,
            document_id=DOCUMENT_ID,
            corpus_version=version,
        )

        print("\nWarming up retrieval...")
        retrieve()

        miss_times = []

        print("\nRunning cache-miss benchmark...")

        for i in range(MISS_RUNS):
            cache_delete(cache_key)
            elapsed = retrieve()
            miss_times.append(elapsed)
            print(f"Miss {i + 1}: {elapsed:.2f} ms")

        hit_times = []

        print("\nRunning cache-hit benchmark...")

        for i in range(HIT_RUNS):
            elapsed = retrieve()
            hit_times.append(elapsed)
            print(f"Hit {i + 1}: {elapsed:.2f} ms")

        report("CACHE MISS RESULTS", miss_times)
        report("CACHE HIT RESULTS", hit_times)

        speedup = mean(miss_times) / mean(hit_times)

        reduction = (
            (mean(miss_times) - mean(hit_times))
            / mean(miss_times)
        ) * 100

        print("\nRETRIEVAL CACHE PERFORMANCE")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Latency reduction: {reduction:.2f}%")

    finally:
        if cache_key:
            cache_delete(cache_key)

        delete_document(
            USER_ID,
            DOCUMENT_ID,
            invalidate_cache=True,
        )


if __name__ == "__main__":
    benchmark()
