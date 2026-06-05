from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            # pyrefly: ignore [missing-import]
            import chromadb

            # Initialize ChromaDB in-memory ephemeral client
            self._chroma_client = chromadb.EphemeralClient()
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []
        
        query_emb = self._embedding_fn(query)
        scored_records = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            scored_records.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        
        # Sort by score descending
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            if ids:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "embeddings"]
            )
            normalized = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                embeddings = results["embeddings"][0]
                for i in range(len(ids)):
                    score = compute_similarity(query_emb, embeddings[i])
                    normalized.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i] or {},
                        "score": score
                    })
            normalized.sort(key=lambda x: x["score"], reverse=True)
            return normalized
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter,
                include=["documents", "metadatas", "embeddings"]
            )
            normalized = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                embeddings = results["embeddings"][0]
                for i in range(len(ids)):
                    score = compute_similarity(query_emb, embeddings[i])
                    normalized.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i] or {},
                        "score": score
                    })
            normalized.sort(key=lambda x: x["score"], reverse=True)
            return normalized
        else:
            filtered_records = []
            for r in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if r.get("metadata", {}).get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(r)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                size_before = self._collection.count()
                # Try deleting by exact ID
                self._collection.delete(ids=[doc_id])
                # Try deleting by metadata filter
                self._collection.delete(where={"doc_id": doc_id})
                size_after = self._collection.count()
                return size_after < size_before
            except Exception:
                return False
        else:
            initial_size = len(self._store)
            self._store = [
                r for r in self._store
                if r["id"] != doc_id and r.get("metadata", {}).get("doc_id") != doc_id
            ]
            return len(self._store) < initial_size
