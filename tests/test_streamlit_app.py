from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

from app.models.api import (
    QueryResponse,
    RerankTrace,
    RetrievalTrace,
    RetrievedChunkTrace,
    SourceChunk,
)
from app.models.enums import Category, DocType
from app.models.metadata import ChunkMetadata

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")

QUERY_RESPONSE = QueryResponse(
    answer="Nettoomsattningen var 150,0 MSEK.",
    sources=[
        SourceChunk(
            raw_text="| Nyckeltal | Varde |\n| Nettoomsattning | 150,0 |",
            metadata=ChunkMetadata(
                tenant_id="tenant_1",
                company_name="SME Corp AB",
                doc_type=DocType.FINANCIAL_REPORT,
                category=Category.FINANCING,
                year=2024,
                quarter=3,
                source_filename="q3_report_2024.pdf",
                object_key="tenant_1/doc-1/q3_report_2024.pdf",
                page_number=1,
                chunk_id="c1",
                previous_chunk_id=None,
                next_chunk_id="c2",
                language="sv",
                processed_timestamp=datetime.now(UTC),
            ),
            score=0.82,
            rerank_score=0.95,
        )
    ],
    trace=RetrievalTrace(
        embedding_model="text-embedding-3-small",
        answer_model="gpt-5-mini",
        filters={"tenant_id": "tenant_1", "year": 2024},
        top_k=5,
        hits=[
            RetrievedChunkTrace(
                chunk_id="c1",
                source_filename="q3_report_2024.pdf",
                page_number=1,
                score=0.82,
                preview="Nettoomsattning 150,0",
                vector_rank=3,
                rerank_score=0.95,
            )
        ],
        neighbor_chunk_ids=["c2"],
        context_chunk_count=2,
        retrieval_ms=120,
        generation_ms=2400,
        reasoning_summary="Jag letade upp tabellen i den finansiella oversikten.",
        rerank=RerankTrace(
            model="rerank-multilingual-v3.0",
            candidates=20,
            applied=True,
            rerank_ms=180,
        ),
    ),
)

SOURCE_URL_RESPONSE = {
    "url": "http://localhost:9000/rag-documents/tenant_1/doc-1/q3_report_2024.pdf?signed",
    "filename": "q3_report_2024.pdf",
    "disposition": "inline",
    "expires_in_seconds": 900,
}


def fake_get(url: str, **kwargs: Any) -> httpx.Response:
    request = httpx.Request("GET", url)
    if url.endswith("/health"):
        payload: dict[str, str] = {"status": "ok", "qdrant": "ok", "minio": "ok"}
        return httpx.Response(200, json=payload, request=request)
    return httpx.Response(200, json=SOURCE_URL_RESPONSE, request=request)


def fake_post(url: str, **kwargs: Any) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(200, json=QUERY_RESPONSE.model_dump(mode="json"), request=request)


def test_chat_renders_answer_trace_and_source_link() -> None:
    with patch("httpx.get", side_effect=fake_get), patch("httpx.post", side_effect=fake_post):
        app = AppTest.from_file(APP_PATH, default_timeout=30)
        app.run()
        assert not app.exception

        app.chat_input[0].set_value("Vad var nettoomsattningen i Q3?").run()
        assert not app.exception

        rendered = " ".join(element.value for element in app.markdown)
        assert "Nettoomsattningen var 150,0 MSEK." in rendered
        assert "Jag letade upp tabellen" in rendered
        captions = " ".join(element.value for element in app.caption)
        assert "rankade om 20 kandidater" in captions
        assert "rerank 0.950" in captions
        links = list(app.get("link_button"))
        assert links and links[0].proto.url == SOURCE_URL_RESPONSE["url"]


def test_rerank_fallback_is_surfaced_as_a_warning() -> None:
    degraded = QUERY_RESPONSE.model_copy(deep=True)
    assert degraded.trace.rerank is not None
    degraded.trace.rerank.applied = False
    degraded.trace.rerank.fallback_reason = "Rerank endpoint returned 401"

    def degraded_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(200, json=degraded.model_dump(mode="json"), request=request)

    with patch("httpx.get", side_effect=fake_get), patch("httpx.post", side_effect=degraded_post):
        app = AppTest.from_file(APP_PATH, default_timeout=30)
        app.run()
        app.chat_input[0].set_value("Vad var nettoomsattningen i Q3?").run()
        assert not app.exception

        warnings = " ".join(element.value for element in app.warning)
        assert "vektorordningen behölls" in warnings
        assert "401" in warnings
