import argparse
import time
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a precision-focused policy document assistant. Follow these rules:
1. EXTRACT exact phrases from context
2. ANSWER ONLY what is asked
3. FORMAT lists clearly
4. CITE sources precisely

Context:
{context}

Question: {question}

Required Answer Format:
[Direct quotation or exact numbers from context]
Sources: [filename.pdf(page)]
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    response = query_rag(args.query_text, verbose=args.verbose)
    latency = time.time() - start_time

    print(f"\nRESPONSE (in {latency:.2f}s):")
    print(response)

def query_rag(query_text: str, verbose: bool = False):
    if verbose:
        print("Prompt ready — calling model now...", flush=True)
        print("Generating response:", flush=True)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    model = OllamaLLM(
        model="phi3:3.8b",
        temperature=0,
        num_ctx=512,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    results = db.similarity_search_with_score(
        query_text + " exact numbers and lists",
        k=1
    )

    context = format_context(results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        question=query_text
    )

    response = model.invoke(prompt)
    return clean_response(response, results)

def format_context(results):
    return "\n\n".join(
        f"FROM {doc.metadata['source']} PAGE {doc.metadata.get('page', 0)}:\n{doc.page_content}"
        for doc, _ in results
    )

def clean_response(response: str, results) -> str:
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if not lines:
        return "Not found in documents"

    sources = sorted({
        f"{doc.metadata['source']}(pg{doc.metadata.get('page', 0)})"
        for doc, _ in results
    })

    return f"{lines[0]}\nSources: {', '.join(sources)}"

if __name__ == "__main__":
    main()
