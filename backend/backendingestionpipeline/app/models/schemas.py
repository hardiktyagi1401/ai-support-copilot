"""
Pydantic schemas for the query/retrieval API layer.

Architecture note:
    We split retrieval schemas from generation schemas deliberately.
    RetrievalResult is the output of pure semantic search — no LLM involved.
    It is the contract for the /query/retrieve endpoint, and it is also
    the *input* to the generation stage when we add it later.

    Keeping this boundary explicit means:
    - We can test retrieval quality independently of generation quality
    - We can swap the generation model without touching the retrieval contract
    - The debug panel gets structured data, not scraped LLM output

    ChunkResult is the atomic unit — one retrieved chunk with its score.
    Everything else composes from it.
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.database import DocumentStatus


# ---------------------------------------------------------------------------
# Ingest schemas (unchanged from before)
# ---------------------------------------------------------------------------

class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    message: str = "Document received and queued for processing"
    model_config = {"from_attributes": True}



class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Retrieval schemas — the core output of semantic search
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    """
    One retrieved chunk from ChromaDB, enriched with similarity score.

    distance vs similarity:
        ChromaDB with cosine space returns a 'distance' value.
        Distance 0.0 = identical vectors (perfect match).
        Distance 2.0 = maximally dissimilar (opposite directions).

        We convert to similarity = 1 - (distance / 2), giving a [0, 1]
        range where 1.0 = perfect match. This is more intuitive for
        threshold decisions and for displaying to users.

    Why store both?
        Raw distance is what ChromaDB gives you. Similarity is what humans
        reason about. Storing both means we can validate our conversion
        formula and compare against other systems' score conventions.
    """
    chunk_id: str = Field(description="ChromaDB vector ID for this chunk")
    document_id: str = Field(description="Links back to PostgreSQL documents table")
    filename: str
    page_number: int | None = None
    chunk_index: int
    text: str = Field(description="Full chunk text — what gets injected into the LLM prompt")
    raw_distance: float = Field(
        description="Raw ChromaDB cosine distance [0, 2]. Lower = more similar."
    )
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Normalised similarity [0, 1]. Higher = more relevant."
    )
    passed_threshold: bool = Field(
        description="Whether this chunk meets the configured score threshold"
    )
    token_count: int | None = None
    embedding_model: str | None = None


class QueryRequest(BaseModel):
    """Incoming retrieval request from any client."""
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of candidate chunks to retrieve before threshold filtering"
    )
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Override the default threshold from settings. Useful for experiments."
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Restrict retrieval to specific documents. None = search all tenant docs."
    )
    tenant_id: str = Field(
        description="Required for multi-tenant isolation. All searches are scoped to this tenant."
    )


class RetrievalResult(BaseModel):
    """
    The complete output of the retrieval stage.

    This is the boundary object between retrieval and generation.
    Generation receives this and decides what to do with it.

    Design decision: we always return ALL top_k chunks with their scores,
    not just the ones that passed the threshold. This lets callers (and the
    debug panel) see what was *almost* retrieved — which is often the most
    informative signal for diagnosing retrieval problems.
    """
    query: str
    query_embedding_ms: int = Field(description="Time to embed the query in milliseconds")
    search_ms: int = Field(description="Time for ChromaDB vector search in milliseconds")
    total_ms: int = Field(description="End-to-end retrieval latency")

    chunks_retrieved: int = Field(description="Total chunks returned by ChromaDB (= top_k)")
    chunks_passed_threshold: int = Field(description="Chunks meeting the score threshold")
    score_threshold_used: float

    chunks: list[ChunkResult] = Field(
        description="All retrieved chunks, sorted by similarity descending"
    )

    embedding_model: str
    tenant_id: str


class InspectRequest(QueryRequest):
    """
    Extended request for the debug/inspect endpoint.
    Inherits all fields from QueryRequest and adds inspection controls.
    """
    include_embedding_preview: bool = Field(
        default=False,
        description="Include first 10 floats of query embedding. Useful for verifying model."
    )
    candidate_multiplier: int = Field(
        default=3, ge=1, le=10,
        description=(
            "Retrieve top_k × multiplier candidates internally, then re-rank and return top_k. "
            "Exposes whether better chunks exist just outside the top_k cutoff."
        )
    )


class ScoreDistribution(BaseModel):
    """Statistical summary of retrieval scores for a given query."""
    min_score: float
    max_score: float
    mean_score: float
    median_score: float
    std_dev: float
    score_gap: float = Field(
        description=(
            "Difference between top score and second score. "
            "Large gap = clear winner. Small gap = ambiguous retrieval."
        )
    )


class InspectResult(BaseModel):
    """
    Full diagnostic output from the inspect endpoint.
    Everything the debug panel needs to evaluate retrieval quality.
    """
    retrieval: RetrievalResult

    # Candidates beyond top_k (if candidate_multiplier > 1)
    extended_candidates: list[ChunkResult] = Field(
        default_factory=list,
        description="Chunks ranked just outside top_k — shows what was almost retrieved"
    )

    score_distribution: ScoreDistribution | None = None
    query_embedding_preview: list[float] | None = Field(
        default=None,
        description="First 10 floats of the query embedding vector"
    )

    # Diagnostic signals
    retrieval_quality_signal: str = Field(
        description=(
            "Human-readable assessment: 'strong', 'moderate', 'weak', or 'no_results'. "
            "Based on top score and score distribution — not LLM-generated."
        )
    )
    diagnostics: list[str] = Field(
        default_factory=list,
        description="Specific observations about this retrieval: gaps, low scores, etc."
    )


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
    request_id: str | None = None




# app/models/schemas.py — add these to your existing file

from pydantic import BaseModel, Field
from typing import Literal
import uuid
from datetime import datetime


class CitationSource(BaseModel):
    """
    One source that contributed to the answer.
    Matches exactly to a retrieved chunk — no invented sources.
    """
    chunk_index: int
    filename: str
    page_number: int | None
    relevance_score: float = Field(ge=0.0, le=1.0)
    text_preview: str = Field(
        description="First 200 chars of the chunk — lets frontend show 'View source'"
    )


class HallucinationFlags(BaseModel):
    """
    Deterministic signals that the answer may not be grounded.
    These are heuristics, not ground truth.
    """
    no_chunks_passed_threshold: bool = False
    answer_contains_no_citation: bool = False
    low_term_overlap: bool = False
    suspiciously_long_answer: bool = False

    @property
    def any_flag_raised(self) -> bool:
        return any([
            self.no_chunks_passed_threshold,
            self.answer_contains_no_citation,
            self.low_term_overlap,
            self.suspiciously_long_answer,
        ])


class RetrievalStats(BaseModel):
    """Retrieval diagnostics passed through to the response."""
    chunks_retrieved: int
    chunks_passed_threshold: int
    top_similarity_score: float | None
    score_threshold_used: float
    embed_ms: int
    search_ms: int


class LatencyBreakdown(BaseModel):
    """Every stage timed. Essential for identifying bottlenecks."""
    retrieval_ms: int
    prompt_assembly_ms: int
    llm_ms: int
    total_ms: int


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    tenant_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = None
    # Conversation history — last N turns for context
    conversation_history: list[dict] | None = Field(
        default=None,
        description="[{'role': 'user'|'assistant', 'content': str}] — last 3-4 turns"
    )


class AskResponse(BaseModel):
    """
    The complete response from the RAG pipeline.

    answered=False means the system intentionally withheld an answer
    because retrieval confidence was too low. The frontend should show
    a graceful refusal message, not an empty answer field.

    answered=True with hallucination_flags.any_flag_raised=True means
    an answer was generated but the system detected warning signals.
    Surface this as a low-confidence indicator in the UI.
    """
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    answered: bool
    answer: str | None = Field(
        description="None when answered=False. Never an empty string."
    )
    refusal_reason: str | None = Field(
        description="Human-readable explanation when answered=False"
    )
    citations: list[CitationSource] = Field(default_factory=list)
    hallucination_flags: HallucinationFlags
    retrieval_stats: RetrievalStats
    latency: LatencyBreakdown
    citation_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of answer sentences containing a [Src N] citation.",
    )
    uncited_sentence_count: int = Field(
        default=0,
        description="Raw count of sentences without citations.",
    )
    model_used: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CitationSource(BaseModel):
    chunk_index: int
    filename: str
    page_number: int | None
    relevance_score: float = Field(ge=0.0, le=1.0)
    text_preview: str
 
 
class HallucinationFlags(BaseModel):
    no_chunks_passed_threshold: bool = False
    answer_contains_no_citation: bool = False
    low_term_overlap: bool = False          # repurposed: low citation coverage < 70%
    suspiciously_long_answer: bool = False
 
    @property
    def any_flag_raised(self) -> bool:
        return any([
            self.no_chunks_passed_threshold,
            self.answer_contains_no_citation,
            self.low_term_overlap,
            self.suspiciously_long_answer,
        ])
 
    def model_dump(self, **kwargs):
        return {
            "no_chunks_passed_threshold": self.no_chunks_passed_threshold,
            "answer_contains_no_citation": self.answer_contains_no_citation,
            "low_term_overlap": self.low_term_overlap,
            "suspiciously_long_answer": self.suspiciously_long_answer,
            "any_flag_raised": self.any_flag_raised,
        }