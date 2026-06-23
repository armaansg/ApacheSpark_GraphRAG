#!/usr/bin/env python3
"""
Graph RAG chat over Jira chunks + module links + ranked catalog code snippets.

Run from project root:
  python3 scripts/graph_rag_chat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expand_context import expand_context, load_catalog_by_module, load_jira_module_indexes
from rag_common import (
    chat_completion,
    hits_for_answer,
    load_index,
    missing_jira_keys,
    primary_seeds,
    retrieve,
)


def build_expanded_context(related_jira, code_snippets) -> str:
    parts = []
    for i, ticket in enumerate(related_jira, 1):
        parts.append(
            f"[Related Jira {i}: {ticket['jira_key']} (module: {ticket['module']})]\n"
            f"{ticket['text']}"
        )
    for i, snippet in enumerate(code_snippets, 1):
        parts.append(
            f"[Code {i}: {snippet['path']} (module: {snippet['module']})]\n"
            f"{snippet['text']}"
        )
    return "\n\n".join(parts)


def answer_graph(query, seed_hits, keys, related_jira, code_snippets):
    seed_parts = []
    for i, hit in enumerate(hits_for_answer(seed_hits, keys), 1):
        seed_parts.append(
            f"[Seed {i}: {hit['path_or_key']}]\n{hit['text']}"
        )
    seed_context = "\n\n".join(seed_parts)
    expanded_context = build_expanded_context(related_jira, code_snippets)

    prompt = f"""You are a helpful assistant for Apache Spark Jira issues and related code.

Answer the question using only the context below. Be accurate and concise.
If the context does not contain enough information, say so clearly.
When citing information, reference SPARK keys or file paths as appropriate.

Primary sources (from semantic search):
{seed_context}
"""
    if expanded_context.strip():
        prompt += f"""
Additional context (from module links and related code):
{expanded_context}
"""

    prompt += f"\nQuestion: {query}\n"
    return chat_completion(prompt)


def answer_missing_tickets(missing: list[str]) -> str:
    if len(missing) == 1:
        return (
            f"{missing[0]} is not in the indexed corpus "
            "(90-day Jira window). I cannot answer questions about it."
        )
    keys = ", ".join(missing)
    return (
        f"These tickets are not in the indexed corpus (90-day Jira window): {keys}. "
        "I cannot answer questions about them."
    )


def main():
    print("Loading index and graph assets...")
    index, by_key = load_index()
    ticket_to_modules, module_to_tickets = load_jira_module_indexes()
    catalog_by_module = load_catalog_by_module()
    print(f"Loaded {len(index)} embedded chunks.")
    print("Type a question (or 'quit' to exit).\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        print("\nRetrieving seed candidates...")
        hits, keys = retrieve(query, index, by_key)
        for hit in hits:
            tag = " [direct key match]" if hit["path_or_key"] in keys else ""
            print(f"  - {hit['path_or_key']}{tag}")

        missing = missing_jira_keys(keys, by_key)
        if keys and missing and not primary_seeds(hits, keys, by_key):
            print(f"\nNot in index: {', '.join(missing)}")
            print("\nAssistant:")
            print(answer_missing_tickets(missing))
            print()
            continue

        seeds = primary_seeds(hits, keys, by_key)
        print(f"\nPrimary seeds for expansion ({len(seeds)}):")
        for seed in seeds:
            print(f"  - {seed['path_or_key']}")

        print("\nGraph expansion...")
        related_jira, code_snippets, expanded_chars = expand_context(
            seeds,
            query,
            keys,
            by_key,
            ticket_to_modules,
            module_to_tickets,
            catalog_by_module,
        )
        for ticket in related_jira:
            print(f"  - {ticket['jira_key']} (module: {ticket['module']})")
        for snippet in code_snippets:
            print(f"  - {snippet['path']} (code snippet)")
        print(
            f"  -> {len(related_jira)} related Jira, "
            f"{len(code_snippets)} code files (~{expanded_chars:,} chars expanded)"
        )

        print("\nAssistant:")
        print(answer_graph(query, hits, keys, related_jira, code_snippets))
        print()


if __name__ == "__main__":
    main()
