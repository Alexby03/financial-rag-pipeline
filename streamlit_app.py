"""Streamlit chat UI for the RAG API, exposing what the agent retrieved and why."""

import os

import httpx
import streamlit as st

from app.models.api import IngestResponse, QueryResponse, SourceChunk, SourceUrlResponse
from app.models.enums import Category, DocType

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default_tenant_test_corp_123")
REQUEST_TIMEOUT = 120.0

st.set_page_config(page_title="Financial RAG", layout="wide")


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
    label = (
        f"Agentens resonemang — {trace.retrieval_ms} ms sökning, "
        f"{trace.generation_ms} ms svar, {trace.context_chunk_count} chunkar i kontext"
    )
    with st.expander(label):
        st.markdown("**1. Sökning i Qdrant**")
        st.caption(f"Embedding-modell `{trace.embedding_model}`, top_k {trace.top_k}")
        st.write({key: str(value) for key, value in trace.filters.items()})

        st.markdown("**2. Omrankning**")
        if trace.rerank is None:
            st.caption("Avstängd — ordningen kommer direkt från vektorsökningen.")
        elif trace.rerank.applied:
            st.caption(
                f"`{trace.rerank.model}` rankade om {trace.rerank.candidates} kandidater "
                f"på {trace.rerank.rerank_ms} ms."
            )
        else:
            st.warning(
                "Omrankningen hoppades över, vektorordningen behölls. "
                f"{trace.rerank.fallback_reason}"
            )

        st.markdown("**3. Träffar och likhet**")
        if trace.hits:
            st.dataframe(
                [
                    {
                        "Vektorplats": hit.vector_rank,
                        "Score": round(hit.score, 4),
                        "Rerank": "—" if hit.rerank_score is None else round(hit.rerank_score, 4),
                        "Källa": hit.source_filename,
                        "Sida": hit.page_number,
                        "Utdrag": hit.preview,
                    }
                    for hit in trace.hits
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Inga chunkar matchade filtren.")

        st.markdown("**4. Grannexpansion**")
        if trace.neighbor_chunk_ids:
            st.caption(
                f"{len(trace.neighbor_chunk_ids)} angränsande chunkar hämtades via "
                "den dubbellänkade listan för att komplettera träffarna."
            )
        else:
            st.caption("Inga grannar hämtades.")

        st.markdown(f"**5. Svar med `{trace.answer_model}`**")
        if trace.reasoning_summary:
            st.markdown(trace.reasoning_summary)
        else:
            st.caption("Modellen svarade utan att behöva resonera i flera steg.")


def render_source(source: SourceChunk, api_url: str, index: int) -> None:
    meta = source.metadata
    score = f"{source.score:.3f}" if source.score is not None else "—"
    if source.rerank_score is not None:
        score = f"{score} · rerank {source.rerank_score:.3f}"
    with st.container(border=True):
        header, action = st.columns([4, 1])
        with header:
            st.markdown(f"**{index}. {meta.source_filename}** — sida {meta.page_number}")
            st.caption(f"Score {score} · {meta.doc_type.value} · {meta.category.value}")
        with action:
            link = fetch_source_url(api_url, meta.tenant_id, meta.object_key)
            if link is None:
                st.caption("Filen saknas i MinIO")
            elif link.disposition == "inline":
                st.link_button(
                    "Öppna",
                    source_link_url(link, meta.page_number),
                    width="stretch",
                )
            else:
                st.link_button("Ladda ner", link.url, width="stretch")
        with st.expander("Visa chunk"):
            st.markdown(source.raw_text)


def render_answer(response: QueryResponse, api_url: str) -> None:
    st.markdown(response.answer)
    render_trace(response)
    if response.sources:
        st.markdown("**Källor**")
        for index, source in enumerate(response.sources, start=1):
            render_source(source, api_url, index)


def run_query(api_url: str, payload: dict[str, object]) -> QueryResponse:
    response = httpx.post(f"{api_url}/query", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return QueryResponse.model_validate(response.json())


def upload_document(
    api_url: str,
    files: dict[str, object],
    data: dict[str, object],
) -> IngestResponse:
    response = httpx.post(f"{api_url}/ingest", files=files, data=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return IngestResponse.model_validate(response.json())


with st.sidebar:
    st.header("Inställningar")
    api_url = st.text_input("API-adress", value=DEFAULT_API_URL).rstrip("/")
    tenant_id = st.text_input("Tenant ID", value=DEFAULT_TENANT_ID)

    st.subheader("Filter")
    doc_type = st.selectbox(
        "Dokumenttyp", options=[None, *DocType], format_func=lambda x: "Alla" if x is None else x
    )
    category = st.selectbox(
        "Kategori", options=[None, *Category], format_func=lambda x: "Alla" if x is None else x
    )
    year = st.number_input("År", min_value=1990, max_value=2100, value=None, step=1)
    quarter = st.selectbox(
        "Kvartal",
        options=[None, 1, 2, 3, 4],
        format_func=lambda x: "Alla" if x is None else f"Q{x}",
    )
    top_k = st.slider("Antal chunkar (top_k)", min_value=1, max_value=20, value=5)
    expand_neighbors = st.checkbox("Hämta angränsande chunkar", value=True)
    use_rerank = st.checkbox("Ranka om med Cohere Rerank", value=True)

    health = check_health(api_url)
    if health is None:
        st.error("Når inte API:t")
    else:
        st.caption(f"Qdrant: {health['qdrant']} · MinIO: {health['minio']}")

chat_tab, upload_tab = st.tabs(["Chatt", "Ladda upp"])

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["content"], api_url)

    question = st.chat_input("Fråga något om dina dokument")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Söker i dokumenten..."):
                try:
                    answer = run_query(
                        api_url,
                        {
                            "question": question,
                            "tenant_id": tenant_id,
                            "doc_type": doc_type.value if doc_type else None,
                            "category": category.value if category else None,
                            "year": int(year) if year else None,
                            "quarter": quarter,
                            "top_k": top_k,
                            "expand_neighbors": expand_neighbors,
                            "rerank": use_rerank,
                        },
                    )
                except httpx.HTTPError as error:
                    st.error(f"Frågan misslyckades: {error}")
                else:
                    render_answer(answer, api_url)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

with upload_tab:
    st.subheader("Ladda upp dokument")
    with st.form("ingest"):
        uploaded = st.file_uploader("Fil", type=["pdf", "png", "jpg", "jpeg", "xlsx", "txt", "md"])
        company_name = st.text_input("Företagsnamn", value="SME Corp AB")
        upload_doc_type = st.selectbox("Dokumenttyp", options=list(DocType), key="upload_doc_type")
        upload_category = st.selectbox("Kategori", options=list(Category), key="upload_category")
        upload_year = st.number_input("År", min_value=1990, max_value=2100, value=2024, step=1)
        upload_quarter = st.selectbox(
            "Kvartal",
            options=[None, 1, 2, 3, 4],
            format_func=lambda x: "—" if x is None else f"Q{x}",
        )
        language = st.text_input("Språk", value="sv")
        submitted = st.form_submit_button("Ladda upp och indexera")

    if submitted:
        if uploaded is None:
            st.warning("Välj en fil först.")
        else:
            with st.spinner("Parsar och indexerar..."):
                try:
                    result = upload_document(
                        api_url,
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data={
                            "tenant_id": tenant_id,
                            "company_name": company_name,
                            "doc_type": upload_doc_type.value,
                            "category": upload_category.value,
                            "year": int(upload_year),
                            "quarter": upload_quarter if upload_quarter else "",
                            "language": language,
                        },
                    )
                except httpx.HTTPError as error:
                    st.error(f"Uppladdningen misslyckades: {error}")
                else:
                    st.success(f"Indexerade {result.chunk_count} chunkar")
                    st.caption(f"Objekt: `{result.object_key}`")
