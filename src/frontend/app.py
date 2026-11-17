"""Streamlit UI for DocuMind – NotebookLM-inspired Chat & Research Workspace.

Provides the visual application shell and main chat workspace connected to DocuMindService.
Supports light and dark themes, multi-session conversation tracking, grounded answer
synthesis with citations, embedded PDF viewing, and resilient error handling.
"""

import base64
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path when running via 'streamlit run'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pypdf  # noqa: E402
import streamlit as st  # noqa: E402

from src.citations.mapper import EvidenceMapper  # noqa: E402
from src.core.models import (  # noqa: E402
    ChatMessage,
    ConversationSession,
    DocumentRecord,
    EvidenceCard,
    ProcessingStatus,
)
from src.frontend.theme import load_theme, save_theme  # noqa: E402
from src.service.app_service import DocuMindService, ServiceResult  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Backend Service Singleton
# ---------------------------------------------------------------------------
@st.cache_resource
def get_service() -> DocuMindService:
    """Instantiate and cache the DocuMind unified application service."""
    return DocuMindService(base_dir="./data")


service = get_service()


# ---------------------------------------------------------------------------
# Theme handling & CSS
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = load_theme()


def _apply_theme(theme: str) -> None:
    """Inject CSS styling tailored for NotebookLM-inspired light/dark aesthetics."""
    if theme == "light":
        bg_main = "#ffffff"
        bg_sidebar = "#f8f9fa"
        bg_card = "#f1f3f4"
        txt_main = "#202124"
        txt_sub = "#5f6368"
        border_col = "#dadce0"
        btn_bg = "#ffffff"
        btn_txt = "#1a73e8"
        accent_blue = "#1a73e8"
        user_msg_bg = "#e8f0fe"
    else:
        bg_main = "#1e1e1e"
        bg_sidebar = "#181818"
        bg_card = "#252526"
        txt_main = "#e8eaed"
        txt_sub = "#9aa0a6"
        border_col = "#3c4043"
        btn_bg = "#2d2d2d"
        btn_txt = "#8ab4f8"
        accent_blue = "#8ab4f8"
        user_msg_bg = "#2a3b5c"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_main};
            color: {txt_main};
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border_col};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {txt_main};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        p, li, span, label {{
            color: {txt_main};
        }}
        .stCaption {{
            color: {txt_sub} !important;
        }}
        /* NotebookLM style cards */
        .documind-card {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }}
        .hero-banner {{
            background: linear-gradient(135deg, {bg_card} 0%, {user_msg_bg} 100%);
            border: 1px solid {border_col};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }}
        /* Buttons */
        div.stButton > button {{
            background-color: {btn_bg};
            color: {btn_txt};
            border: 1px solid {border_col};
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        div.stButton > button:hover {{
            border-color: {accent_blue};
            color: {accent_blue};
        }}
        /* Chat messages */
        [data-testid="stChatMessage"] {{
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
        }}
        .stTextInput > div > input {{
            background-color: {bg_card};
            color: {txt_main};
            border: 1px solid {border_col};
            border-radius: 8px;
        }}
        .badge-source {{
            display: inline-block;
            background-color: {bg_card};
            color: {accent_blue};
            border: 1px solid {border_col};
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.78rem;
            margin-right: 6px;
            margin-top: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
def _init_active_session() -> str:
    """Ensure at least one conversation session exists and return its ID."""
    summaries = service.list_sessions()
    if not summaries:
        res = service.create_session(title="Notebook 1")
        if res.success and res.data:
            return res.data.session_id
        return ""
    # Use existing active session if valid, or default to most recent
    active = service.get_active_session()
    if active:
        return active.session_id
    service.set_active_session(summaries[0].session_id)
    return summaries[0].session_id


if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = _init_active_session()

if "pdf_view" not in st.session_state:
    st.session_state.pdf_view = None


# ---------------------------------------------------------------------------
# Sidebar Navigation & Session Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 DocuMind")
    st.caption("Grounded Document Intelligence")

    # New Chat action
    if st.button("➕ New Conversation", use_container_width=True):
        count = len(service.list_sessions()) + 1
        new_res = service.create_session(title=f"Notebook {count}")
        if new_res.success and new_res.data:
            st.session_state.active_session_id = new_res.data.session_id
            service.set_active_session(new_res.data.session_id)
            st.session_state.pdf_view = None
            st.rerun()

    # Session Selector
    sessions = service.list_sessions()
    if sessions:
        session_map = {s.session_id: s.title for s in sessions}
        curr_idx = 0
        if st.session_state.active_session_id in session_map:
            curr_idx = list(session_map.keys()).index(st.session_state.active_session_id)

        selected_id = st.selectbox(
            "Active Notebook",
            options=list(session_map.keys()),
            format_func=lambda sid: session_map.get(sid, sid),
            index=curr_idx,
            key="session_selector",
        )
        if selected_id != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_id
            service.set_active_session(selected_id)
            st.session_state.pdf_view = None
            st.rerun()

    st.markdown("---")

    # Navigation
    nav_page = st.radio(
        "Navigation",
        options=["Home", "Documents", "Conversations", "Settings"],
        index=0,
        key="nav_page",
    )

    st.markdown("---")

    # Theme toggle
    theme_choice = st.radio(
        "Theme",
        options=["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1,
        key="theme_selector",
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        save_theme(theme_choice)
        st.rerun()

    # System Status badge
    status = service.get_system_status()
    st.caption(
        f"🤖 Ollama: `llama3.2:3b`\n\n"
        f"📑 Documents: {status['completed_documents']} | Chunks: {status['vector_store_chunks']}"
    )

# Apply styling
_apply_theme(st.session_state.theme)


# ---------------------------------------------------------------------------
# Reusable UI Helpers
# ---------------------------------------------------------------------------
def _format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp into human readable relative string."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return iso_str[:16]


def _extract_evidence_cards(msg: ChatMessage) -> list[EvidenceCard]:
    """Retrieve or reconstruct structured EvidenceCards for an assistant message."""
    if msg.evidence_cards:
        return msg.evidence_cards
    if msg.sources:
        try:
            mapper = EvidenceMapper()
            emap = mapper.map_evidence(
                answer=msg.content,
                retrieved_chunks=msg.sources,
                query=msg.standalone_query,
            )
            if emap.evidence_cards:
                return emap.evidence_cards
        except Exception:
            pass

    # Fallback: construct cards from verified citations and sources
    cards: list[EvidenceCard] = []
    for idx, cit in enumerate(msg.citations, start=1):
        score = 0.0
        tot_pages = 1
        for s in msg.sources:
            if s.chunk_id == cit.chunk_id or (s.doc_id == cit.doc_id and s.page == cit.page):
                score = s.score
                tot_pages = int(s.metadata.get("total_pages", 1))
                break
        cards.append(
            EvidenceCard(
                citation_index=idx,
                doc_id=cit.doc_id,
                filename=cit.filename,
                page=cit.page,
                total_pages=tot_pages,
                chunk_id=cit.chunk_id,
                snippet=cit.snippet,
                similarity_score=score,
                is_verified=cit.is_verified,
                raw_reference=cit.raw_reference,
            )
        )
    return cards


def _extract_pdf_page_text(pdf_path: Path, page_number: int) -> str | None:
    """Safely extract plain text from a specific 1-based page of a PDF."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        if 1 <= page_number <= len(reader.pages):
            return reader.pages[page_number - 1].extract_text()
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# PDF Viewer Panel
# ---------------------------------------------------------------------------
def _render_pdf_viewer_panel() -> None:
    """Render the embedded PDF viewer & citation inspection panel.

    Identifies documents strictly by doc_id and stored file_path.
    Gracefully handles missing documents, missing files, and invalid page numbers.
    """
    view_state = st.session_state.get("pdf_view")
    if not view_state:
        return

    doc_id = view_state.get("doc_id", "")
    req_page = view_state.get("page", 1)
    snippet = view_state.get("snippet", "")
    score = view_state.get("similarity_score", None)
    cit_idx = view_state.get("citation_index", None)

    # Header with title and Close button
    col_hdr, col_close = st.columns([0.78, 0.22])
    with col_close:
        if st.button(
            "✖ Close",
            key="btn_close_pdf_viewer",
            help="Close PDF viewer",
            use_container_width=True,
        ):
            st.session_state.pdf_view = None
            st.rerun()

    # Look up document record by doc_id (NEVER by filename alone)
    doc_rec = service.get_document(doc_id) if doc_id else None
    if not doc_rec:
        with col_hdr:
            st.markdown("### 📄 Document Viewer")
        st.warning(
            "⚠️ **Document Unavailable**: The referenced document record was not "
            "found in the registry (it may have been deleted)."
        )
        return

    with col_hdr:
        st.markdown(f"### 📄 {doc_rec.filename}")
        st.caption(f"ID: `{doc_rec.doc_id[:8]}…` | Total Pages: {doc_rec.total_pages}")

    # Check file exists on disk
    pdf_path = Path(doc_rec.file_path)
    if not pdf_path.is_file():
        st.error(
            f"⚠️ **File Missing**: The PDF file for '{doc_rec.filename}' is not found "
            "in storage. It may have been moved or removed from disk."
        )
        return

    # Validate and clamp page numbers gracefully
    total_pages = doc_rec.total_pages or 1
    page = req_page
    if page < 1:
        st.warning(f"⚠️ Page {req_page} is invalid. Displaying Page 1.")
        page = 1
    elif page > total_pages:
        st.warning(
            f"⚠️ Page {req_page} exceeds document length ({total_pages} pages). "
            f"Displaying Page {total_pages}."
        )
        page = total_pages

    # Page navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([0.25, 0.5, 0.25])
    with nav_col1:
        if st.button("◀ Prev", key="btn_prev_page", disabled=(page <= 1), use_container_width=True):
            st.session_state.pdf_view["page"] = page - 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align: center; font-weight: 600; padding-top: 6px;'>"
            f"Page {page} of {total_pages}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button(
            "Next ▶",
            key="btn_next_page",
            disabled=(page >= total_pages),
            use_container_width=True,
        ):
            st.session_state.pdf_view["page"] = page + 1
            st.rerun()

    # Active Citation Context Banner (if opened from a citation)
    if snippet:
        score_badge = f" • Relevance: **{score:.2f}**" if score is not None else ""
        cit_marker = f"[{cit_idx}] " if cit_idx else ""
        st.markdown(
            f"""
            <div class="documind-card" style="margin-top: 4px; margin-bottom: 10px;">
                <span class="badge-source">{cit_marker}Cited on Page {req_page}{score_badge}</span>
                <p style="margin-top: 6px; font-size: 0.88rem; font-style: italic;">
                    "{snippet}"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Extracted page text preview
    page_text = _extract_pdf_page_text(pdf_path, page)
    if page_text:
        with st.expander(f"📝 Page {page} Extracted Text", expanded=False):
            st.text(page_text)

    # Render embedded PDF viewer via iframe
    try:
        pdf_bytes = pdf_path.read_bytes()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <iframe
                src="data:application/pdf;base64,{b64_pdf}#page={page}"
                width="100%"
                height="620px"
                style="border: 1px solid #dadce0; border-radius: 8px; margin-top: 6px;"
            ></iframe>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"⚠️ Could not load PDF preview: {e}")


# ---------------------------------------------------------------------------
# Page: Home / Chat Workspace (NotebookLM Style)
# ---------------------------------------------------------------------------
def _render_chat_workspace() -> None:
    """Main conversational workspace for document question answering."""
    sess_id = st.session_state.active_session_id
    sess_res = service.get_session(sess_id) if sess_id else ServiceResult(success=False)

    if not sess_res.success or not sess_res.data:
        st.error("Active conversation could not be loaded. Please create a new notebook.")
        return

    session: ConversationSession = sess_res.data

    # Top Notebook Header
    col_title, col_actions = st.columns([0.75, 0.25])
    with col_title:
        st.markdown(f"### 💬 {session.title}")
        doc_count = len(session.selected_doc_ids)
        if doc_count > 0:
            st.caption(f"Scoped to {doc_count} selected document(s)")
        else:
            st.caption("Searching across all ingested documents")

    with col_actions:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🧹 Clear", help="Clear chat history", use_container_width=True):
                service.clear_session_messages(session.session_id)
                st.session_state.pdf_view = None
                st.rerun()
        with col_btn2:
            all_docs = service.list_documents()
            st.caption(f"📚 {len(all_docs)} in store")

    st.markdown("---")

    # Responsive layout: split when PDF viewer is open
    has_active_viewer = bool(st.session_state.get("pdf_view"))
    if has_active_viewer:
        col_chat, col_viewer = st.columns([0.55, 0.45], gap="large")
    else:
        col_chat = st.container()
        col_viewer = None

    # ---------------- Chat Column ----------------
    with col_chat:
        messages = session.messages
        if not messages:
            st.markdown(
                """
                <div class="hero-banner">
                    <h3>Welcome to your DocuMind Workspace</h3>
                    <p>DocuMind synthesizes grounded, cited answers from your knowledge base.
                    Ask complex questions, synthesize multiple sources, and trace every fact
                    back to page-accurate citations and interactive PDF previews.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Suggested Inquiries")
            col1, col2 = st.columns(2)
            suggested_prompt = None
            with col1:
                if st.button(
                    "📌 What are the main topics and takeaways?",
                    use_container_width=True,
                ):
                    suggested_prompt = "What are the main topics and takeaways?"
                if st.button(
                    "🔍 Explain the system architecture and flow",
                    use_container_width=True,
                ):
                    suggested_prompt = "Explain the system architecture and flow."
            with col2:
                if st.button(
                    "⚡ What are the key performance metrics?",
                    use_container_width=True,
                ):
                    suggested_prompt = "What are the key performance metrics?"
                if st.button(
                    "❓ Compare the advantages and tradeoffs",
                    use_container_width=True,
                ):
                    suggested_prompt = "Compare the advantages and tradeoffs discussed."

            if suggested_prompt:
                _handle_user_query(suggested_prompt, session)

        else:
            # Render message timeline
            for m_idx, msg in enumerate(messages):
                if msg.role == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg.content)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg.content)

                        # Metadata & latency indicators
                        meta_parts = []
                        if msg.latency_seconds is not None:
                            meta_parts.append(f"⏱️ {msg.latency_seconds:.2f}s")
                        if msg.standalone_query and msg.standalone_query != msg.content:
                            meta_parts.append(f"🔎 Rewritten: *'{msg.standalone_query}'*")
                        if not msg.is_grounded:
                            meta_parts.append("⚠️ General fallback / insufficient context")

                        if meta_parts:
                            st.caption(" • ".join(meta_parts))

                        # Evidence Cards & Citation Navigation
                        cards = _extract_evidence_cards(msg)
                        if cards:
                            expander_label = f"📑 Evidence & Citations ({len(cards)})"
                            with st.expander(expander_label, expanded=True):
                                for card in cards:
                                    score_str = f"{card.similarity_score:.2f}"
                                    header_md = (
                                        f"**[{card.citation_index}] {card.filename}** "
                                        f"(Page {card.page}) • *Relevance: {score_str}*"
                                    )
                                    st.markdown(header_md)
                                    if card.snippet:
                                        st.caption(f"> \"{card.snippet}\"")

                                    # Interactive button to open PDF viewer at cited page
                                    btn_key = (
                                        f"open_card_{msg.message_id}_{card.citation_index}_"
                                        f"{card.doc_id}_{card.page}_{m_idx}"
                                    )
                                    if st.button(
                                        f"📄 Open in PDF (Page {card.page})",
                                        key=btn_key,
                                    ):
                                        st.session_state.pdf_view = {
                                            "doc_id": card.doc_id,
                                            "page": card.page,
                                            "snippet": card.snippet,
                                            "similarity_score": card.similarity_score,
                                            "citation_index": card.citation_index,
                                        }
                                        st.rerun()
                                    st.markdown("---")

    # ---------------- Viewer Column ----------------
    if col_viewer is not None:
        with col_viewer:
            _render_pdf_viewer_panel()

    # Chat Input Box at Bottom
    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        _handle_user_query(user_query, session)


def _handle_user_query(query: str, session: ConversationSession) -> None:
    """Execute conversational turn via DocuMindService."""
    cleaned = query.strip()
    if not cleaned:
        st.warning("Please enter a question before sending.")
        return

    # Render immediate user message bubble
    with st.chat_message("user", avatar="👤"):
        st.markdown(cleaned)

    # Invoke LLM RAG synthesis with loading indicator
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Synthesizing answer from documents..."):
            result: ServiceResult = service.ask(
                query=cleaned,
                session_id=session.session_id,
            )

        if result.success and result.data:
            resp = result.data
            st.markdown(resp.answer)

            # Auto-title conversation on first turn if default title
            if len(session.messages) == 0 and session.title.startswith("Notebook"):
                truncated_title = cleaned[:28] + ("..." if len(cleaned) > 28 else "")
                service.update_session_title(session.session_id, truncated_title)

            st.rerun()
        else:
            # Display user-friendly structured error
            err_msg = result.error_message or "An unexpected error occurred."
            if result.error_code == "OLLAMA_CONNECTION_ERROR":
                st.error(
                    f"⚠️ **Ollama Service Unavailable**\n\n{err_msg}\n\n"
                    "Please verify local Ollama is running (`ollama serve`)."
                )
            elif result.error_code == "EMPTY_QUERY":
                st.warning("Query cannot be empty.")
            else:
                st.error(f"❌ **Error**: {err_msg}")


# ---------------------------------------------------------------------------
# Page: Documents Library
# ---------------------------------------------------------------------------
def _render_documents_page() -> None:
    """Document library with upload, deletion, and conversation scope selection."""
    st.markdown("## 📑 Document Library")
    st.markdown("---")

    # -------------------------------------------------
    # PDF Upload (multiple files, unique filenames)
    # -------------------------------------------------
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    if "upload_alerts" not in st.session_state:
        st.session_state.upload_alerts = []

    # Display pending upload alerts
    if st.session_state.upload_alerts:
        for alert_type, alert_text in st.session_state.upload_alerts:
            if alert_type == "success":
                st.success(alert_text)
            else:
                st.error(alert_text)
        st.session_state.upload_alerts = []

    if uploaded_files:
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        alerts = []

        for uploaded_file in uploaded_files:
            # Store each upload in its own UUID-named subdirectory so the
            # original filename is preserved as path.name (what the registry
            # stores) while two uploads with the same filename never collide
            # on disk.
            slot_dir = upload_dir / uuid.uuid4().hex
            slot_dir.mkdir(parents=True, exist_ok=True)
            dest_path = slot_dir / uploaded_file.name
            dest_path.write_bytes(uploaded_file.getbuffer())

            try:
                with st.spinner(f"Ingesting `{uploaded_file.name}` …"):
                    res = service.ingest_document(dest_path)

                if res.success:
                    alerts.append(("success", f"✅ `{uploaded_file.name}` ingested successfully."))
                else:
                    # Clean up the temp file if ingestion failed
                    dest_path.unlink(missing_ok=True)
                    if slot_dir.exists():
                        slot_dir.rmdir()
                    err_detail = f"❌ Failed to ingest `{uploaded_file.name}`: {res.error_message}"
                    alerts.append(("error", err_detail))
            except Exception as exc:
                # Ensure slot_dir and dest_path are cleaned up even on unexpected exceptions
                dest_path.unlink(missing_ok=True)
                if slot_dir.exists():
                    slot_dir.rmdir()
                err_msg = f"❌ Unexpected ingestion error for `{uploaded_file.name}`: {exc}"
                alerts.append(("error", err_msg))

        st.session_state.upload_alerts = alerts
        st.rerun()

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    docs: list[DocumentRecord] = service.list_documents()
    col_total, col_ready, col_chunks = st.columns(3)
    with col_total:
        st.metric("Total Documents", len(docs))
    with col_ready:
        ready = sum(1 for d in docs if d.status == ProcessingStatus.COMPLETED)
        st.metric("Indexed & Ready", ready)
    with col_chunks:
        status_info = service.get_system_status()
        st.metric("Vector Chunks", status_info["vector_store_chunks"])

    if not docs:
        st.info("No documents have been ingested yet. Use the upload control above to add PDFs.")
        return

    # -------------------------------------------------
    # Document Table
    # -------------------------------------------------
    st.markdown("### Document Index")
    table_rows = []
    for doc in docs:
        table_rows.append(
            {
                "ID (short)": doc.doc_id[:8] + "…",
                "Filename": doc.filename,
                "Uploaded": _format_timestamp(doc.upload_timestamp),
                "Pages": doc.total_pages,
                "Status": doc.status.value.upper(),
            }
        )
    st.table(table_rows)

    # -------------------------------------------------
    # Document Deletion
    # -------------------------------------------------
    st.markdown("### Delete a Document")
    del_doc_id = st.selectbox(
        "Select a document to delete",
        options=[d.doc_id for d in docs],
        format_func=lambda did: next(
            (d.filename for d in docs if d.doc_id == did), did[:12]
        ),
        key="delete_selector",
    )
    if st.button("🗑️ Delete Selected Document"):
        del_res = service.delete_document(del_doc_id)
        if del_res.success:
            st.success("Document deleted successfully.")
            # Clear viewer state if viewing the deleted document
            if (
                st.session_state.pdf_view
                and st.session_state.pdf_view.get("doc_id") == del_doc_id
            ):
                st.session_state.pdf_view = None
        else:
            st.error(f"Deletion failed: {del_res.error_message}")
        st.rerun()

    # -------------------------------------------------
    # Document Scope for Active Conversation
    # -------------------------------------------------
    st.markdown("---")
    st.markdown("### Conversation Document Scope")
    st.caption(
        "Select which documents the active conversation should search. "
        "Leave empty to search all ingested documents."
    )

    completed_docs = [d for d in docs if d.status == ProcessingStatus.COMPLETED]
    if not completed_docs:
        st.info("No completed documents are available to add to a conversation.")
        return

    # Restore previously persisted selection from the active session
    sess_res = service.get_session(st.session_state.active_session_id)
    persisted_ids: list[str] = []
    if sess_res.success and sess_res.data:
        persisted_ids = [
            did for did in sess_res.data.selected_doc_ids
            if any(d.doc_id == did for d in completed_docs)
        ]

    selected_ids = st.multiselect(
        "Documents in scope",
        options=[d.doc_id for d in completed_docs],
        format_func=lambda did: next(
            (d.filename for d in completed_docs if d.doc_id == did), did[:12]
        ),
        default=persisted_ids,
        key="doc_scope_selector",
    )
    if st.button("Update Conversation Scope"):
        upd_res = service.update_session_documents(
            st.session_state.active_session_id, selected_ids
        )
        if upd_res.success:
            st.success("Conversation document scope updated and persisted.")
        else:
            st.error(f"Failed to update scope: {upd_res.error_message}")


# ---------------------------------------------------------------------------
# Page: Conversations Manager
# ---------------------------------------------------------------------------
def _render_conversations_page() -> None:
    """Manage past conversations, session switching, and deletion."""
    st.markdown("## 💬 Conversation Sessions")
    st.markdown("---")

    sessions = service.list_sessions()
    if not sessions:
        st.info("No conversation sessions found.")
        return

    st.markdown(f"Found **{len(sessions)}** active conversation session(s):")

    for s in sessions:
        with st.container():
            is_active = s.session_id == st.session_state.active_session_id
            badge = " ⭐ (Active)" if is_active else ""

            col_info, col_act1, col_act2 = st.columns([0.65, 0.20, 0.15])
            with col_info:
                st.markdown(f"#### 📓 {s.title}{badge}")
                st.caption(
                    f"Messages: {s.message_count} | "
                    f"Created: {_format_timestamp(s.created_at)} | "
                    f"Updated: {_format_timestamp(s.updated_at)}"
                )
            with col_act1:
                if not is_active:
                    if st.button("Switch", key=f"switch_{s.session_id}"):
                        st.session_state.active_session_id = s.session_id
                        service.set_active_session(s.session_id)
                        st.session_state.pdf_view = None
                        st.rerun()
                else:
                    st.success("Currently Active")
            with col_act2:
                if st.button("🗑️ Delete", key=f"del_{s.session_id}"):
                    service.delete_session(s.session_id)
                    st.session_state.active_session_id = _init_active_session()
                    st.session_state.pdf_view = None
                    st.rerun()

            st.markdown("---")


# ---------------------------------------------------------------------------
# Page: Settings & Diagnostics
# ---------------------------------------------------------------------------
def _render_settings_page() -> None:
    """System settings, model parameters, and storage diagnostics."""
    st.markdown("## ⚙️ Settings & Diagnostics")
    st.markdown("---")

    st.markdown("### System Architecture")
    diag = service.get_system_status()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="documind-card">
                <h4>🧠 LLM & Embeddings</h4>
                <p><strong>Chat Model:</strong> <code>llama3.2:3b</code></p>
                <p><strong>Embedding Model:</strong> <code>nomic-embed-text</code></p>
                <p><strong>Vector Dimensions:</strong> 768</p>
                <p><strong>Ollama Base URL:</strong> <code>http://localhost:11434</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="documind-card">
                <h4>💾 Storage & Persistence</h4>
                <p><strong>SQLite Registry:</strong> <code>./data/registry.db</code></p>
                <p><strong>Chroma Vectorstore:</strong> <code>./data/vectorstore</code></p>
                <p><strong>Conversations Store:</strong> <code>./data/conversations</code></p>
                <p><strong>Total Vector Chunks:</strong> {diag['vector_store_chunks']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Application Theme")
    st.write(f"Active Theme: **{st.session_state.theme.capitalize()}**")
    st.caption("Switch themes using the toggle in the left navigation sidebar.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if nav_page == "Home":
    _render_chat_workspace()
elif nav_page == "Documents":
    _render_documents_page()
elif nav_page == "Conversations":
    _render_conversations_page()
elif nav_page == "Settings":
    _render_settings_page()

# Global Footer
st.markdown("---")
st.caption("DocuMind • Grounded Document Question Answering • Local Ollama Engine")
