from query_data import query_rag
import time
import statistics
import sys
import difflib
from typing import List

# Set to True to log similarity scores and debug matching
VERBOSE = False

TEST_CASES = [
    {
        "question": "What are the top five courses in Kenyan TVET institutions?",
        "expected": ["tailoring", "engineering", "masonry", "carpentry", "hair", "beauty"],
        "max_latency": 600.0
    },
    {
        "question": "How many TVET institutions exist across 47 counties?",
        "expected": ["2,313"],
        "max_latency": 300.0
    },
    {
        "question": "Why do most Kenyan youth pursue TVET training instead of continuing with formal education?",
        "expected": [
            "dropping out", "school fees", "teenage pregnancies",
            "loss of interest", "acquired all the education"
        ],
        "max_latency": 180.0
    }
]

def contains_keyword(response: str, keyword: str, threshold: float = 0.5) -> bool:
    response = response.lower()
    keyword = keyword.lower()

    # Fast path: exact match
    if keyword in response:
        return True

    window_size = len(keyword.split()) + 2
    words = response.split()

    for i in range(len(words) - window_size + 1):
        chunk = " ".join(words[i:i + window_size])
        ratio = difflib.SequenceMatcher(None, chunk, keyword).ratio()
        if VERBOSE:
            print(f"[DEBUG] Chunk: '{chunk}' vs Keyword: '{keyword}' -> Score: {ratio:.2f}")
        if ratio >= threshold:
            return True

    return False

def assert_response(response: str, expected: List[str], latency: float, max_latency: float) -> bool:
    missing = [kw for kw in expected if not contains_keyword(response, kw)]
    if missing:
        print(f"Missing keywords: {missing}", file=sys.stderr, flush=True)
        return False

    if "Sources:" not in response:
        print("Missing source citations", file=sys.stderr, flush=True)
        return False

    if latency > max_latency:
        print(f"Latency {latency:.2f}s exceeds SLA of {max_latency}s", file=sys.stderr, flush=True)
        return False

    return True

def run_benchmark():
    results = []
    print("Starting RAG System Benchmark", flush=True)

    for test in TEST_CASES:
        print(f"\n{'=' * 80}", flush=True)
        print(f"Test: {test['question']}", flush=True)

        start = time.time()
        try:
            response = query_rag(test["question"], verbose=True)
        except Exception as e:
            print(f"\nError during test: {str(e)}", file=sys.stderr, flush=True)
            response = ""

        latency = time.time() - start

        print(f"\nRESPONSE (in {latency:.2f}s):", flush=True)
        print(response, flush=True)

        passed = assert_response(
            response=response,
            expected=test["expected"],
            latency=latency,
            max_latency=test["max_latency"]
        )

        results.append({
            "question": test["question"],
            "passed": passed,
            "latency": latency
        })

        print(f"\nTest Result:", flush=True)
        print(f"Passed: {'PASS' if passed else 'FAIL'}", flush=True)
        print(f"Latency: {latency:.2f}s (Max allowed: {test['max_latency']}s)", flush=True)

    print(f"\n{'=' * 80}", flush=True)
    avg_latency = statistics.mean(r['latency'] for r in results)
    pass_rate = sum(r['passed'] for r in results) / len(results)

    print(f"SUMMARY:", flush=True)
    print(f"Average latency: {avg_latency:.2f}s", flush=True)
    print(f"Pass rate: {pass_rate * 100:.0f}%", flush=True)
    print(f"✔️ Successful tests: {sum(r['passed'] for r in results)}/{len(results)}", flush=True)

if __name__ == "__main__":
    run_benchmark()

