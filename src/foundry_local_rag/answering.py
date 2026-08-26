"""Application-level composition of local retrieval, grounding, and chat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chat import FoundryLocalChatAdapter
from .embeddings import FoundryLocalEmbeddingAdapter
from .prompting import GroundedPrompt, InsufficientContext, prepare_grounded_prompt
from .retrieval import RetrievedChunk, retrieve_chunks

TOP_K = 3


@dataclass(frozen=True)
class AnswerResult:
    """One answer and the exact retrieved chunks supplied as its evidence."""

    answer: str
    sources: tuple[RetrievedChunk, ...]


class AnswerOrchestrator:
    """Answer questions through caller-owned local adapters."""

    def __init__(
        self,
        database_path: Path,
        embedding_adapter: FoundryLocalEmbeddingAdapter,
        chat_adapter: FoundryLocalChatAdapter,
    ) -> None:
        self._database_path = Path(database_path)
        self._embedding_adapter = embedding_adapter
        self._chat_adapter = chat_adapter

    def answer(self, question: str) -> AnswerResult:
        """Retrieve evidence, prepare grounded input, and return an answer."""

        retrieved_chunks = retrieve_chunks(
            question,
            self._database_path,
            self._embedding_adapter,
            TOP_K,
        )
        prepared = prepare_grounded_prompt(question, retrieved_chunks)
        if isinstance(prepared, InsufficientContext):
            return AnswerResult(answer=prepared.answer, sources=())

        assert isinstance(prepared, GroundedPrompt)
        answer = self._chat_adapter.complete(prepared.messages)
        return AnswerResult(answer=answer, sources=prepared.sources)
