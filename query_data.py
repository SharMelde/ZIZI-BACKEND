from rag import get_answer

def query_rag(question: str, verbose: bool = False) -> str:
    if verbose:
        print(f"[query_rag] Question received: {question}")
    answer = get_answer(question)
    if verbose:
        print(f"[query_rag] Answer generated: {answer}")
    return answer

