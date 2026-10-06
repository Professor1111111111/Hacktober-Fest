import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        self.texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # BM25
        tokenized = [
            text.lower().split()
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(tokenized)

        # Embeddings
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True
        )

        self.embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        # FAISS
        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(self.embeddings)

    def search(self, query, top_k=5):

        # BM25
        bm25_scores = self.bm25.get_scores(
            query.lower().split()
        )

        # FAISS
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        faiss_scores, faiss_indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks))
        )

        # Combine results
        scores = {}

        for i, score in enumerate(bm25_scores):
            scores[i] = float(score)

        for score, index in zip(
            faiss_scores[0],
            faiss_indices[0]
        ):
            scores[index] = (
                scores.get(index, 0)
                + float(score)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            self.chunks[index]
            for index, score in ranked[:top_k]
        ]