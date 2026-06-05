from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentence_endings = r'(?<=\. |\! |\? |\.\n)'
        raw_sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        chunks = []
        for j in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[j : j + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            chunks = []
            for start in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[start : start + self.chunk_size])
            return chunks

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        chunks = []
        current_chunk = []
        current_len = 0

        for part in splits:
            if not part:
                continue

            if len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                sub_chunks = self._split(part, next_separators)
                chunks.extend(sub_chunks)
            else:
                sep_len = len(separator) if current_chunk else 0
                if current_len + sep_len + len(part) > self.chunk_size:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = [part]
                    current_len = len(part)
                else:
                    current_chunk.append(part)
                    current_len += sep_len + len(part)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_prod = _dot(vec_a, vec_b)
    norm_a = sum(x * x for x in vec_a)
    norm_b = sum(x * x for x in vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_prod / (math.sqrt(norm_a) * math.sqrt(norm_b))


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=int(chunk_size * 0.1))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        strategies = {
            "fixed_size": fixed_chunker,
            "by_sentences": sentence_chunker,
            "recursive": recursive_chunker
        }

        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0

            comparison[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }

        return comparison
