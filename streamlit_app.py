"""Streamlit chat UI for the RAG API, exposing what the agent retrieved and why."""

import os
import time

import httpx
import streamlit as st

from app.models.api import (
    IngestJobAccepted,
    IngestJobStatusResponse,
    QueryResponse,
    SourceChunk,
    SourceUrlResponse,
)
from app.models.enums import Category, DocType

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default_tenant_test_corp_123")
REQUEST_TIMEOUT = 120.0
INGEST_POLL_INTERVAL_SECONDS = 1.5
INGEST_POLL_TIMEOUT_SECONDS = 1800.0

STAGE_LABELS = {
    "checking": "Checking for duplicates...",
    "uploading": "Uploading to storage...",
    "parsing": "Parsing the document...",
    "embedding": "Creating embeddings...",
    "indexing": "Indexing in Qdrant...",
}

st.set_page_config(page_title="Financial RAG", layout="wide")
st.markdown(
    """
    <style>
    /* Streamlit's toolbar cycles people/sport icons while a script runs; use a CSS spinner. */
    [data-testid="stStatusWidgetRunningManIcon"] { display: none !important; }
    [data-testid="stStatusWidgetRunningIcon"] {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    [data-testid="stStatusWidgetRunningIcon"]::before {
      content: "";
      width: 1rem;
      height: 1rem;
      border: 2px solid currentColor;
      border-right-color: transparent;
      border-radius: 50%;
      animation: rag-spin 0.7s linear infinite;
    }
    @keyframes rag-spin {
      to { transform: rotate(360deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_source_url(api_url: str, tenant_id: str, object_key: str) -> SourceUrlResponse | None:
    """Presigned MinIO link so the browser fetches the file without proxying it through the API."""
    try:
        response = httpx.get(
            f"{api_url}/sources/url",
            params={"object_key": object_key, "tenant_id": tenant_id},
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return SourceUrlResponse.model_validate(response.json())


def source_link_url(link: SourceUrlResponse, page_number: int) -> str:
    """The #page fragment stays in the browser, so it never invalidates the MinIO signature."""
    if link.disposition == "inline" and link.filename.lower().endswith(".pdf"):
        return f"{link.url}#page={page_number}"
    return link.url


def check_health(api_url: str) -> dict[str, str] | None:
    try:
        response = httpx.get(f"{api_url}/health", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.json()


def render_trace(response: QueryResponse) -> None:
    trace = response.trace
    vision_part = f", {trace.vision_ms} ms vision" if trace.figure_readings else ""
    label = (
        f"Agent reasoning — {trace.retrieval_ms} ms retrieval{vision_part}, "
        f"{trace.generation_ms} ms answer, {trace.context_chunk_count} chunks in context"
    )
    with st.expander(label):
        st.markdown("**1. Qdrant search**")
        st.caption(f"Embedding model `{trace.embedding_model}`, top_k {trace.top_k}")
        st.write({key: str(value) for key, value in trace.filters.items()})

        st.markdown("**2. Reranking and fusion**")
        if trace.rerank is None:
            st.caption("Cohere rerank: off.")
        elif trace.rerank.applied:
            st.caption(
                f"`{trace.rerank.model}` reranked {trace.rerank.candidates} candidates "
                f"in {trace.rerank.rerank_ms} ms."
            )
        else:
            st.warning(
                "Reranking was skipped; vector order was kept. "
                f"{trace.rerank.fallback_reason}"
            )
        if trace.fusion is not None and trace.fusion.lexical_applied:
            lists = "Cohere + BM25" if trace.fusion.cohere_applied else "vector + BM25"
            st.caption(
                f"RRF fusion ({lists}, k={trace.fusion.rrf_k}) over "
                f"{trace.fusion.candidates} candidates."
            )

        st.markdown("**3. Hits and similarity**")
        if trace.hits:
            st.dataframe(
                [
                    {
                        "Vector": hit.vector_rank,
                        "Lexical": "—" if hit.lexical_rank is None else hit.lexical_rank,
                        "RRF": "—" if hit.rrf_score is None else round(hit.rrf_score, 4),
                        "Score": round(hit.score, 4),
                        "Rerank": "—" if hit.rerank_score is None else round(hit.rerank_score, 4),
                        "Source": hit.source_filename,
                        "Page": hit.page_number,
                        "Preview": hit.preview,
                    }
                    for hit in trace.hits
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No chunks matched the filters.")

        st.markdown("**4. Neighbor expansion**")
        if trace.neighbor_chunk_ids:
            st.caption(
                f"{len(trace.neighbor_chunk_ids)} neighboring chunks were fetched via "
                "the doubly linked list to complement the hits."
            )
        else:
            st.caption("No neighbors were fetched.")

        st.markdown("**5. Figure vision**")
        if trace.figure_readings:
            applied = sum(1 for reading in trace.figure_readings if reading.applied)
            model_note = f" `{trace.vision_model}`" if trace.vision_model else ""
            st.caption(
                f"{applied} of {len(trace.figure_readings)} retrieved figures were read"
                f"{model_note} in {trace.vision_ms} ms."
            )
            for reading in trace.figure_readings:
                heading = f"{reading.source_filename} p.{reading.page_number}"
                if reading.applied:
                    with st.expander(heading, expanded=True):
                        st.markdown(reading.reading)
                else:
                    st.warning(f"{heading}: {reading.error}")
        else:
            st.caption(
                "No figure crops were read — either no figure chunks were retrieved, "
                "or `OPENAI_VISION_MODEL` is blank."
            )

        st.markdown(f"**6. Answer with `{trace.answer_model}`**")
        if trace.reasoning_summary:
            st.markdown(trace.reasoning_summary)
        else:
            st.caption("The model answered without multi-step reasoning.")


def render_source(source: SourceChunk, api_url: str, index: int) -> None:
    meta = source.metadata
    score = f"{source.score:.3f}" if source.score is not None else "—"
    if source.rerank_score is not None:
        score = f"{score} · rerank {source.rerank_score:.3f}"
    with st.container(border=True):
        header, action = st.columns([4, 1])
        with header:
            label = "figure" if meta.figure_object_key else "page"
            st.markdown(
                f"**{index}. {meta.source_filename}** — {label} {meta.page_number}"
            )
            st.caption(f"Score {score} · {meta.doc_type.value} · {meta.category.value}")
        with action:
            link = fetch_source_url(api_url, meta.tenant_id, meta.object_key)
            if link is None:
                st.caption("File missing in MinIO")
            elif link.disposition == "inline":
                st.link_button(
                    "Open",
                    source_link_url(link, meta.page_number),
                    width="stretch",
                )
            else:
                st.link_button("Download", link.url, width="stretch")
        with st.expander("Show chunk"):
            st.markdown(source.raw_text)


def chat_history_payload(messages: list[dict[str, object]]) -> list[dict[str, str]]:
    """Prior turns for /query — assistant content is the answer text only."""
    history: list[dict[str, str]] = []
    for message in messages:
        role = str(message["role"])
        if role == "user":
            history.append({"role": "user", "content": str(message["content"])})
            continue
        content = message["content"]
        if isinstance(content, QueryResponse):
            history.append({"role": "assistant", "content": content.answer})
        else:
            parsed = QueryResponse.model_validate(content)
            history.append({"role": "assistant", "content": parsed.answer})
    return history


def as_query_response(content: object) -> QueryResponse:
    if isinstance(content, QueryResponse):
        return content
    return QueryResponse.model_validate(content)


def render_answer(response: QueryResponse, api_url: str) -> None:
    st.markdown(response.answer)
    render_trace(response)
    if response.sources:
        label = f"Sources — {len(response.sources)}"
        with st.expander(label):
            for index, source in enumerate(response.sources, start=1):
                render_source(source, api_url, index)


def run_query(api_url: str, payload: dict[str, object]) -> QueryResponse:
    response = httpx.post(f"{api_url}/query", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return QueryResponse.model_validate(response.json())


def submit_ingest(
    api_url: str,
    files: dict[str, object],
    data: dict[str, object],
) -> IngestJobAccepted:
    response = httpx.post(f"{api_url}/ingest", files=files, data=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return IngestJobAccepted.model_validate(response.json())


def fetch_ingest_job(api_url: str, job_id: str) -> IngestJobStatusResponse:
    response = httpx.get(f"{api_url}/ingest/jobs/{job_id}", timeout=30.0)
    response.raise_for_status()
    return IngestJobStatusResponse.model_validate(response.json())


def toast_ingest_progress(job: IngestJobStatusResponse) -> None:
    if job.status == "queued":
        st.toast("Upload queued...", icon=":material/progress_activity:", duration="infinite")
        return
    if job.status == "running":
        label = STAGE_LABELS.get(job.stage or "", "Processing the document...")
        st.toast(label, icon=":material/progress_activity:", duration="infinite")
        return
    if job.status == "completed":
        st.toast("Document uploaded", icon=":material/check_circle:", duration="infinite")
        return
    if job.status == "duplicate":
        st.toast("Document was already indexed", icon=":material/info:", duration="infinite")
        return
    st.toast(
        f"Upload failed: {job.error or 'unknown error'}",
        icon=":material/error:",
        duration="infinite",
    )


def wait_for_ingest_job(api_url: str, job_id: str) -> IngestJobStatusResponse:
    """Poll until the lifespan worker finishes; toasts stay until dismissed or replaced."""
    deadline = time.monotonic() + INGEST_POLL_TIMEOUT_SECONDS
    last_key: tuple[str, str | None] | None = None
    while True:
        job = fetch_ingest_job(api_url, job_id)
        key = (job.status, job.stage)
        if key != last_key:
            toast_ingest_progress(job)
            last_key = key
        if job.status in {"completed", "failed", "duplicate"}:
            return job
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException(
                f"Ingest job {job_id} still {job.status} after "
                f"{int(INGEST_POLL_TIMEOUT_SECONDS)}s"
            )
        time.sleep(INGEST_POLL_INTERVAL_SECONDS)


with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    tenant_id = st.text_input("Tenant ID", value=DEFAULT_TENANT_ID)

    st.subheader("Filters")
    company_name = st.text_input(
        "Company name",
        value="",
        placeholder="Optional exact metadata match",
        help="Hard Qdrant filter. Leave empty to auto-scope from the question text.",
    )
    doc_type = st.selectbox(
        "Document type", options=[None, *DocType], format_func=lambda x: "All" if x is None else x
    )
    category = st.selectbox(
        "Category", options=[None, *Category], format_func=lambda x: "All" if x is None else x
    )
    year = st.number_input("Year", min_value=1990, max_value=2100, value=None, step=1)
    quarter = st.selectbox(
        "Quarter",
        options=[None, 1, 2, 3, 4],
        format_func=lambda x: "All" if x is None else f"Q{x}",
    )
    top_k = st.slider("Number of chunks (top_k)", min_value=1, max_value=20, value=5)
    expand_neighbors = st.checkbox("Fetch neighboring chunks", value=True)
    use_rerank = st.checkbox("Rerank with Cohere Rerank", value=True)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    with st.spinner("Checking API..."):
        health = check_health(api_url)
    if health is None:
        st.error("Cannot reach the API")
    else:
        st.caption(f"Qdrant: {health['qdrant']} · MinIO: {health['minio']}")

chat_tab, upload_tab = st.tabs(["Chat", "Upload"])

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(as_query_response(message["content"]), api_url)

    question = st.chat_input("Ask something about your documents")
    if question:
        prior_history = chat_history_payload(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    answer = run_query(
                        api_url,
                        {
                            "question": question,
                            "tenant_id": tenant_id,
                            "company_name": company_name.strip() or None,
                            "doc_type": doc_type.value if doc_type else None,
                            "category": category.value if category else None,
                            "year": int(year) if year else None,
                            "quarter": quarter,
                            "top_k": top_k,
                            "expand_neighbors": expand_neighbors,
                            "rerank": use_rerank,
                            "chat_history": prior_history,
                        },
                    )
                except httpx.HTTPError as error:
                    st.error(f"Query failed: {error}")
                    st.session_state.messages.pop()
                else:
                    render_answer(answer, api_url)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

with upload_tab:
    st.subheader("Upload document")
    with st.form("ingest"):
        uploaded = st.file_uploader(
            "File",
            type=[
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "docx",
                "pptx",
                "xlsx",
                "txt",
                "md",
                "sie",
                "si",
                "se",
                "xhtml",
                "html",
                "htm",
            ],
        )
        company_name = st.text_input("Company name", value="", placeholder="Auto from document")
        upload_doc_type = st.selectbox(
            "Document type",
            options=[None, *DocType],
            format_func=lambda x: "Auto" if x is None else x,
            key="upload_doc_type",
        )
        upload_category = st.selectbox(
            "Category",
            options=[None, *Category],
            format_func=lambda x: "Auto" if x is None else x,
            key="upload_category",
        )
        upload_year = st.number_input("Year", min_value=1990, max_value=2100, value=None, step=1)
        upload_quarter = st.selectbox(
            "Quarter",
            options=[None, 1, 2, 3, 4],
            format_func=lambda x: "Auto" if x is None else f"Q{x}",
        )
        language = st.text_input("Language", value="sv")
        submitted = st.form_submit_button("Upload and index")

    if submitted:
        if uploaded is None:
            st.warning("Choose a file first.")
        else:
            with st.spinner("Queueing the upload..."):
                try:
                    ingest_data: dict[str, object] = {
                        "tenant_id": tenant_id,
                        "language": language,
                    }
                    if company_name.strip():
                        ingest_data["company_name"] = company_name.strip()
                    if upload_doc_type is not None:
                        ingest_data["doc_type"] = upload_doc_type.value
                    if upload_category is not None:
                        ingest_data["category"] = upload_category.value
                    if upload_year:
                        ingest_data["year"] = int(upload_year)
                    if upload_quarter:
                        ingest_data["quarter"] = upload_quarter
                    accepted = submit_ingest(
                        api_url,
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data=ingest_data,
                    )
                    result = wait_for_ingest_job(api_url, accepted.job_id)
                except httpx.HTTPError as error:
                    st.error(f"Upload failed: {error}")
                else:
                    if result.status == "failed":
                        st.error(f"Upload failed: {result.error or 'unknown error'}")
                    elif result.status == "duplicate":
                        st.info(
                            f"The file is already indexed with {result.chunk_count} chunks. "
                            "No new parsing was done."
                        )
                        st.caption(f"Object: `{result.object_key}`")
                    else:
                        st.success(f"Indexed {result.chunk_count} chunks")
                        st.caption(f"Object: `{result.object_key}`")
