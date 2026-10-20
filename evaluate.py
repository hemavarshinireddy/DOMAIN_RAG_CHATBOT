"""Automatic evaluation sheet.  Run from the project root:  python tests/evaluate.py

Needs: an index built from your documents (use the app once) and tests/test_questions.csv.
Leave expected_source blank (or write 'Not available') for questions that must be refused.
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import vector_store  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402


def main():
    store = vector_store.load_saved()
    if store is None:
        sys.exit("No saved index found. Run the app, upload PDFs and click 'Process Documents' first.")
    pipe = RAGPipeline(store)

    with open(ROOT / "tests" / "test_questions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results, ans_total, ans_hit, una_total, una_hit, times = [], 0, 0, 0, 0, []
    for row in rows:
        q = row["question"].strip()
        exp_src = (row.get("expected_source") or "").strip()
        exp_page = (row.get("expected_page") or "").strip()
        unavailable = exp_src == "" or exp_src.lower() == "not available"

        t0 = time.time()
        res = pipe.answer(q)
        dt = time.time() - t0
        times.append(dt)
        retrieved = ", ".join(f"{s} p{p}" for s, p in res["retrieved"]) or "-"

        if unavailable:
            una_total += 1
            ok = not res["found"]
            una_hit += ok
        else:
            ans_total += 1
            ok = res["found"] and any(
                s == exp_src and (not exp_page or str(p) == exp_page) for s, p in res["retrieved"]
            )
            ans_hit += ok

        results.append({
            "question": q,
            "expected_source": exp_src or "Not available",
            "expected_page": exp_page,
            "retrieved_sources": retrieved,
            "answer": res["answer"].replace("\n", " ")[:300],
            "correct": "YES" if ok else "NO",
            "seconds": round(dt, 2),
        })
        print(f"[{'OK' if ok else 'XX'}] {q}  ({dt:.1f}s)")

    out = ROOT / "tests" / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n=== SUMMARY ===")
    if ans_total:
        print(f"Retrieval + answer accuracy : {ans_hit}/{ans_total} ({100*ans_hit/ans_total:.0f}%)")
    if una_total:
        print(f"Correct refusals            : {una_hit}/{una_total} ({100*una_hit/una_total:.0f}%)")
    print(f"Average response time       : {sum(times)/len(times):.1f}s")
    print(f"Detailed sheet saved to     : {out}")


if __name__ == "__main__":
    main()
