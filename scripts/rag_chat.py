import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_common import answer_plain, load_index, retrieve

def main():
    print("Loading index...")
    index, by_key = load_index()
    print(f"Loaded {len(index)} embedded chunks.")
    print("Type a question (or 'quit' to exit).\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        print("\nRetrieving...")
        hits, keys = retrieve(query, index, by_key)
        for hit in hits:
            tag = " [direct key match]" if hit["path_or_key"] in keys else ""
            print(f"  - {hit['path_or_key']}{tag}")
        print("\nAssistant:")
        print(answer_plain(query, hits, keys))
        print()
        
if __name__ == "__main__":
    main()