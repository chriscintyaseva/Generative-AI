from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str]) -> np.ndarray:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts
        }
    )

    response.raise_for_status()

    data = response.json()

    vecs = np.array(
        data["embeddings"],
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vecs,
        axis=1,
        keepdims=True
    )

    return vecs / np.where(norms == 0, 1, norms)


@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(
        default=None,
        repr=False
    )


class FilteredVectorStore:

    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(
        self,
        docs: list[FilteredDocument]
    ) -> None:

        embeddings = embed_texts(
            [d.text for d in docs]
        )

        for doc, emb in zip(
            docs,
            embeddings
        ):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[
            Callable[[FilteredDocument], bool]
        ] = None
    ) -> list[tuple[FilteredDocument, float]]:
        """Search with optional metadata filter applied before ranking."""

        candidates = (
            self._docs
            if filter_fn is None
            else [
                d for d in self._docs
                if filter_fn(d)
            ]
        )

        if not candidates:
            return []

        q_vec = embed_texts([query])[0]

        matrix = np.array(
            [d.embedding for d in candidates],
            dtype=np.float32
        )

        scores = matrix @ q_vec

        k = min(
            k,
            len(candidates)
        )

        top_idx = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                candidates[int(i)],
                float(scores[i])
            )
            for i in top_idx
        ]


docs = [
    FilteredDocument(
        "a1",
        "GPT-4o supports vision and function calling.",
        {
            "category": "openai",
            "year": 2024
        }
    ),
    FilteredDocument(
        "a2",
        "Claude 3.5 Sonnet excels at coding tasks.",
        {
            "category": "anthropic",
            "year": 2024
        }
    ),
    FilteredDocument(
        "a3",
        "GPT-4o-mini is a smaller, cheaper model.",
        {
            "category": "openai",
            "year": 2024
        }
    ),
    FilteredDocument(
        "a4",
        "Claude Opus 4 is Anthropic's most capable model.",
        {
            "category": "anthropic",
            "year": 2025
        }
    ),
    FilteredDocument(
        "a5",
        "GPT-4 Turbo has a 128K context window.",
        {
            "category": "openai",
            "year": 2023
        }
    )
]


fstore = FilteredVectorStore()

fstore.add(docs)


print("=== All docs ===")

results = fstore.search(
    "which model is good at coding?",
    k=3
)

for doc, score in results:
    print(
        f" [{score:.4f}] "
        f"{doc.id}: {doc.text}"
    )


print("\n=== Anthropic only ===")

results = fstore.search(
    "which model is good at coding?",
    k=3,
    filter_fn=lambda d: d.metadata["category"] == "anthropic"
)

for doc, score in results:
    print(
        f" [{score:.4f}] "
        f"{doc.id}: {doc.text}"
    )