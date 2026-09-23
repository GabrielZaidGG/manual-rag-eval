from rag_eval_project.generate_ans import answer_question
from rag_eval_project.evalqueries import EVAL_QUERIES


def looks_like_refusal(answer: str) -> bool:
    """
    Detect refusal-like answers based on observed phrasing.
    Case-insensitive check.
    """
    refusal_phrases = [
        "the context provided does not contain",
        "i cannot answer the question based on the given context",
        "does not contain",                                          
        "is not in the context",
        "cannot answer",
        "does not mention"
    ]
    ans_lower = answer.lower()
    return any(phrase in ans_lower for phrase in refusal_phrases)


def run_eval(eval_queries: list[dict]) -> list[dict]:
    results = []
    for item in eval_queries:
        output = answer_question(item["query"])
        actual = "refusal" if looks_like_refusal(output["answer"]) else "grounded_answer"
        results.append({
            "query": item["query"],
            "expected": item["expect"],
            "actual": actual,
            "passed": actual == item["expect"],
            "answer": output["answer"],
        })
    return results


if __name__ == "__main__":
    eval_queries = EVAL_QUERIES

    results = run_eval(eval_queries)
    for r in results:
        print(r)