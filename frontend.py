import hashlib
import html
import os
import re
import tempfile
import traceback
import uuid
from pathlib import Path

import streamlit as st

from src.add_documents import add_files
from src.rag_advanced.chain import get_rag_chain
from src.rag_advanced.components import classify_query
from src.rag_advanced.utils import RAGMode, set_verbose, detect_language
from src.vector_store import get_document_content, get_retriever

K_DOCS = 10

# --- DIALOGS ---
@st.dialog("📄 Document Content", width="large")
def show_document(project, inr, section):
    # CSS for the document viewer
    st.markdown("""
        <style>
        .doc-container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: 'Georgia', serif;
            color: #333;
            line-height: 1.6;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .doc-header {
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .doc-title {
            font-size: 1.4em;
            font-weight: bold;
            color: #1a1a1a;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
        }
        .doc-meta {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
            font-family: 'Segoe UI', sans-serif;
        }
        .doc-content {
            font-size: 1.05em;
            white-space: pre-wrap; /* Preserve formatting but wrap */
        }
        .highlight {
            background-color: #fffbdd;
            padding: 0 4px;
            border-radius: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.spinner("Retrieving document..."):
        content = get_document_content(project, inr, section)

    # Clean up content slightly for display
    # Replace the "---" separator logic from vector_store with a visual separator
    clean_content = content.replace("\n\n---\n\n", "<hr class='doc-separator'>")

    # Escape user content to prevent XSS
    safe_project = html.escape(project)
    safe_inr = html.escape(inr)
    safe_section = html.escape(section)
    safe_content = html.escape(clean_content).replace("\n", "<br>")

    html_block = f"""
    <div class="doc-container">
        <div class="doc-header">
            <div class="doc-title">{safe_project}</div>
            <div class="doc-meta">
                <span style="margin-right: 15px;"><strong>INR:</strong> {safe_inr}</span>
                <span><strong>Section:</strong> {safe_section}</span>
            </div>
        </div>
        <div class="doc-content">
            {safe_content}
        </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


st.set_page_config(page_title="SGIA Intelligence", page_icon="⚡", layout="centered")


def file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

from src.topics_bertopic import load_bertopic, merge_intent_and_grounded_topics, top_topics_for_query, suggest_questions_from_topics, topics_from_retrieved_chunks
@st.cache_resource
def load_topic_model():
    return load_bertopic("output/bertopic_model.pkl")

@st.cache_resource
def load_chain(k_docs: int, mode: str, with_summary: bool):
    """Load retriever + chain only once per process."""
    retriever = get_retriever(k_docs=k_docs)
    # Map string mode to Enum
    rag_mode = RAGMode(mode)
    chain = get_rag_chain(retriever, mode=rag_mode, k_docs=k_docs, with_summary=with_summary)
    return chain, retriever


st.title("⚡ SGIA Intelligence")
st.caption("133 Texas Interconnection Agreements")


def keep_only_last_sources(text: str) -> str:
    """
    If there are multiple 'Sources:' blocks, keep only the last one.
    Everything after the last 'Sources:' stays, any earlier 'Sources:' blocks are removed.
    """
    last = text.rfind("Sources:")
    if last == -1:
        return text

    # Keep answer before last sources + keep last sources block
    prefix = text[:last].strip()
    last_block = text[last:].strip()

    # Remove any earlier 'Sources:' occurrences in the prefix by truncating them
    # (i.e., answer should not contain any 'Sources:' at all)
    first_in_prefix = prefix.find("Sources:")
    if first_in_prefix != -1:
        prefix = prefix[:first_in_prefix].strip()

    return (prefix + "\n\n" + last_block).strip()


# -------------------------
# Sidebar: Settings + Upload
# -------------------------
with st.sidebar:
    logo_path = Path(__file__).parent / "ercot_image.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.header("OPTIONS")

    # Mode selection
    def reset_conversation():
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

    mode_options = [m.value for m in RAGMode]
    selected_mode = st.radio(
        "Select Mode",
        mode_options,
        index=0,
        format_func=lambda x: x.capitalize(),
        key="rag_mode_selection",
        on_change=reset_conversation
    )

    with_summary = st.toggle("Auto-summarization", value=False)
    show_verbose = st.checkbox("Show internal processing", value=False)
    enable_bertopic = False  # BERTopic disabled

    k_docs = st.slider("Number of retrieved documents", 1, 50, K_DOCS)

    if st.button("New chat"):
        st.session_state.clear()
        st.rerun()

    st.divider()

    # --- state ---
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "uploaded_docs" not in st.session_state:
        # Docs that have been indexed (or at least accepted into the uploaded section)
        # list[dict]: {"name": str, "size": int, "saved_path": str, "chunks": int}
        st.session_state.uploaded_docs = []

    st.subheader("ADD DOCUMENTS")

    selected_files = st.file_uploader(
        "Upload PDF/TXT/MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    file_label = st.text_input("File label (optional)", value="")

    index_clicked = st.button("Index documents", disabled=not selected_files)

    if index_clicked and selected_files:
        # 1) Save selected files to disk
        saved_paths = []
        file_ids = []
        original_names = []

        for uf in selected_files:
            original_names.append(uf.name)
            suffix = Path(uf.name).suffix.lower()  # .pdf, .txt, .md
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            saved_paths.append(tmp_path)
            fid = file_hash(tmp_path)
            file_ids.append(fid)

        extra_metadata = {}
        if file_label.strip():
            extra_metadata["project_name"] = file_label.strip()

        try:
            # 2) Index them
            stats = add_files.upsert_documents_to_chroma(
                paths=saved_paths,
                original_names=original_names,
                extra_metadata=extra_metadata if extra_metadata else None
            )

            st.success(f"✅ Indexing completed. Chunks added: {sum(stats.values())}")

            # 3) Move to "Uploaded documents" list (after successful indexing)
            # (we store per-file; chunks returned is total, so we set total or '-' per file)
            for uf, path, fid in zip(selected_files, saved_paths, file_ids):
                st.session_state.uploaded_docs.append({
                    "name": uf.name,
                    "size": uf.size,
                    "saved_path": path,
                    "source_file_id": fid,
                    "chunks": stats.get(path, 0),  # optional: if you later compute per-file chunks, set it here
                })

            # 4) Reset uploader so selected files disappear from Drag&Drop area
            st.session_state.uploader_key += 1

            # 5) Reload retriever/chain so new docs are used
            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")

        finally:
            for p in saved_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

    st.divider()
    st.markdown("**Uploaded documents**")
    if not st.session_state.uploaded_docs:
        st.caption("No uploaded documents yet.")
    else:
        for i, d in enumerate(st.session_state.uploaded_docs):
            c1, c2 = st.columns([8, 2])
            with c1:
                st.write(f"📄 {d['name']}  *(chunks: {d.get('chunks','-')})*")
            with c2:
                if st.button("🗑️", key=f"del_indexed_{i}", help="Remove from ChromaDB"):
                    try:
                        add_files.delete_document_from_chroma(d["source_file_id"])
                        st.toast("Deleted from index 🗑️", icon="🗑️")

                        st.session_state.uploaded_docs.pop(i)

                        st.cache_resource.clear()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Delete failed: {e}")

# -------------------------
# Load chain (cached)
# -------------------------
chain, retriever = load_chain(k_docs=k_docs, mode=selected_mode, with_summary=with_summary)

# Session id for RunnableWithMessageHistory (src/chat_history.py)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# UI messages (only for rendering)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper to parse and render structured content
def render_message_structurally(content: str, msg_index: int, topics: list = None, questions: list = None):
    """
    Parses the full assistant message to separate:
    1. Main Response
    2. Sources (put in expander as buttons)
    3. Summary (put in distinct block)
    """

    # 1. Extract Summary if present
    summary_split = re.split(r'\n\n--- (?:Summary|Resumen) ---\n', content, maxsplit=1)

    main_and_sources = summary_split[0]
    summary_content = summary_split[1] if len(summary_split) > 1 else None

    # 2. Extract Sources from the first part
    sources_split = main_and_sources.split("Sources:\n", 1)

    main_response = sources_split[0].strip()
    sources_content = sources_split[1].strip() if len(sources_split) > 1 else None

    # --- RENDER ---

    # 1. Main Response - clean up HTML artifacts that break markdown table rendering
    # Only remove specific HTML tags that shouldn't appear in markdown
    clean_response = main_response
    # Remove <br> variants (common LLM artifact in tables)
    clean_response = re.sub(r'<br\s*/?>', ' ', clean_response, flags=re.IGNORECASE)
    # Remove other common HTML tags but NOT < or > alone (preserve math like "<5 MW")
    clean_response = re.sub(r'</?(?:span|div|p|b|i|u|em|strong)[^>]*>', '', clean_response, flags=re.IGNORECASE)
    # Normalize multiple spaces
    clean_response = re.sub(r'  +', ' ', clean_response)

    # --- CURRENCY FORMATTING FIX ---
    # Fix spaces in numbers: "15, 368, 453" -> "15,368,453"
    # Apply repeatedly until no more changes (handles multiple occurrences like "23, 990, 000")
    prev = None
    while prev != clean_response:
        prev = clean_response
        # Match digit, comma, any whitespace (including non-breaking space), digit
        clean_response = re.sub(r'(\d),[\s\u00a0]+(\d)', r'\1,\2', clean_response)

    # Add space before parenthesis with number: "453(47" -> "453 (47"
    clean_response = re.sub(r'(\d)\((\d)', r'\1 (\2', clean_response)

    # Escape $ signs to prevent LaTeX math interpretation in Streamlit
    # $100 -> \$100 (but preserve already escaped \$)
    clean_response = re.sub(r'(?<!\\)\$', r'\\$', clean_response)

    st.markdown(clean_response)

    # 2. Sources (Expander with Buttons)
    if sources_content:
        with st.expander("📚 Sources"):
            # Split lines and render buttons
            lines = sources_content.strip().split('\n')
            for idx, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line: continue

                # Create unique key for this button
                btn_key = f"src_{msg_index}_{idx}"

                if st.button(clean_line, key=btn_key):
                    # Parse metadata: [1] Project (INR) - Section
                    # Regex: find optional [N], then Project, (INR), -, Section
                    match = re.search(r'(?:\[\d+\]\s*)?(.*?)\s*\((.*?)\)\s*-\s*(.*)', clean_line)
                    if match:
                        project = match.group(1).strip()
                        inr = match.group(2).strip()
                        section = match.group(3).strip()
                        show_document(project, inr, section)
                    else:
                        st.error(f"Could not parse source metadata: {clean_line}")

    if topics or questions:
        with st.expander("🧠 Suggested topics & follow-up questions", expanded=False):
            if topics:
                st.markdown("**🧩 Suggested topics**")
                for t in topics:
                    kws = ", ".join([w for (w, _) in t.get("keywords", [])[:6]])
                    st.write(f"- Topic {t['topic_id']} (score={t['score']:.2f}): {kws if kws else '(no keywords)'}")
            else:
                st.caption("No topic suggestions available.")

            if questions:
                st.markdown("**❓ Suggested questions**")
                for q in questions:
                    st.write(f"- {q}")

    # 3. Summary
    if summary_content:
        st.divider()
        st.subheader("📋 Document Summary")
        st.info(summary_content)

# Render history
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if "logs" in m and m["logs"]:
            with st.status("Internal Processing", expanded=False, state="complete") as status_container:
                for log_msg in m["logs"]:
                     status_container.markdown(log_msg)

        if m["role"] == "assistant":
            render_message_structurally(m["content"], i, m.get("topics"), m.get("followups", []))
        else:
            st.markdown(m["content"])


# Chat input
user_text = st.chat_input("Type your question...")

def parse_log_msg(msg: str) -> str:
    """Helper to style log messages for storage/rendering."""
    if "=====" in msg:
        return None

    if "[STEP]" in msg:
        clean_msg = msg.split("[STEP]", 1)[1].strip()
        return f"**🔄 {clean_msg}**"
    elif "[OK]" in msg:
        clean_msg = msg.split("[OK]", 1)[1].strip()
        # Special pretty print for metadata
        if clean_msg.startswith("Extracted metadata: {"):
            try:
                import ast
                # Extract dict string part
                dict_str = clean_msg.split(":", 1)[1].strip()
                meta_dict = ast.literal_eval(dict_str)
                if meta_dict and isinstance(meta_dict, dict):
                    formatted_lines = [f":green[✓ Extracted metadata:]"]
                    for k, v in meta_dict.items():
                        formatted_lines.append(f"  - **{k}**: `{v}`")
                    return "\n".join(formatted_lines)
            except (ValueError, SyntaxError, TypeError):
                pass  # Fall through to default formatting
        return f":green[✓ {clean_msg}]"
    elif "[WARN]" in msg:
        clean_msg = msg.split("[WARN]", 1)[1].strip()
        return f"⚠️ {clean_msg}" # Using text emoji to be safe in markdown strings or st.warning content?
        # Actually in stored logs we might just store the pre-formatted markdown string.
        # But we can't call st.warning() in a loop easily without creating widgets?
        # Actually st.status supports markdown. formatting as markdown is better.
    elif "[INFO]" in msg:
        clean_msg = msg.split("[INFO]", 1)[1].strip()
        if clean_msg.strip().startswith("Query"):
            return f"- {clean_msg}"
        else:
            return f"_{clean_msg}_" # Italics for info/caption equivalent
    else:
        return msg

if user_text:
    topics = []
    questions = []

    # 1) Render user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) Classify query and display badge
    query_type = classify_query(user_text)
    if query_type == "aggregation":
        st.caption("📊 Using corpus analytics")
    elif query_type == "hybrid":
        st.caption("📊 Using analytics + 📄 document retrieval")
    else:
        st.caption("📄 Using document retrieval")

    # 3) Generate response (streaming)
    with st.chat_message("assistant"):

        # Verbose logging area
        status_container = None
        current_logs = [] # Store formatted logs to save later

        if show_verbose:
            status_container = st.status("Internal Processing", expanded=True)

            def logger_callback(msg):
                """Thread-safe logger callback - only stores messages, doesn't write to Streamlit"""
                formatted_msg = parse_log_msg(msg)
                if formatted_msg:
                    # Only store for later rendering - don't write to Streamlit from thread
                    current_logs.append(formatted_msg)

            set_verbose(True, callback=logger_callback)
        else:
            set_verbose(False)

        placeholder = st.empty()
        full = ""

        try:
            for chunk in chain.stream(
                {"question": user_text},
                config={"configurable": {"session_id": st.session_state.session_id}},
            ):
                full += str(chunk)
                placeholder.markdown(full)


            # Render accumulated logs to status container (thread-safe)
            if status_container and current_logs:
                for log_msg in current_logs:
                    status_container.markdown(log_msg)
                status_container.update(label="Internal Processing Complete", state="complete", expanded=False)

            # Final render with structure (swapping raw stream for pretty UI)
            placeholder.empty()

            # 4) Topic suggestions (skip if out-of-scope)
            # Check if response is an out-of-scope rejection
            # Check for key phrases from both language versions
            is_out_of_scope = (
                "no está relacionada con" in full.lower() or
                "not related to ERCOT" in full.lower() or
                "solo puedo responder preguntas sobre proyectos de energía" in full.lower() or
                "I can only answer questions about energy projects" in full.lower()
            )

            # Debug: log detection status
            if is_out_of_scope:
                st.caption("🔍 Out-of-scope question detected - skipping topic suggestions")

            if not is_out_of_scope and enable_bertopic:
                try:
                    # Detect language for multilingual topic suggestions
                    query_lang = detect_language(user_text)

                    topic_model = load_topic_model()
                    topics_chunks = topics_from_retrieved_chunks(topic_model, retriever, user_text, top_n=3)
                    topics_query = top_topics_for_query(topic_model, user_text, top_n=3)
                    topics = merge_intent_and_grounded_topics(topics_query, topics_chunks, top_n=3)
                    questions = suggest_questions_from_topics(topics, n=5, lang=query_lang)

                except Exception as e:
                    # Optional: show a tiny debug message
                    st.caption(f"Topic suggestions unavailable: {e}")

            formatted_full = keep_only_last_sources(full)

            render_message_structurally(formatted_full, len(st.session_state.messages), topics, questions)

        except Exception as e:
            tb = traceback.format_exc()
            st.error("❌ Error generating response")
            st.code(tb)  # muestra stacktrace completo en la UI
            # opcional: también lo pones en el chat
            full = f"❌ Error generating response: {repr(e)}"
            placeholder.markdown(full)
            if status_container:
                status_container.update(label="Internal Processing Failed", state="error")

        # Reset verbose to avoid side effects
        set_verbose(False)

    # 5) Save assistant response in UI history (including logs for replay)
    formatted_full = keep_only_last_sources(full)

    st.session_state.messages.append({
        "role": "assistant",
        "content": formatted_full,
        "topics": topics,
        "followups": questions,
        "logs": current_logs if show_verbose else [],
    })
