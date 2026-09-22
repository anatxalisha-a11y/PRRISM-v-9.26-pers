import streamlit as st
from urllib.parse import urlparse

from search_engine import load_artifacts, search


st.set_page_config(page_title="FAISS Search", layout="wide")
st.title("RAG / FAISS Query Interface")

@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    index, chunks = get_artifacts()
except Exception as e:
    st.error(f"Could not load artifacts: {e}")
    st.stop()


query = st.text_input("Enter your query")
top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

if st.button("Search") and query.strip():
    try:
        hits = search(index, chunks, query.strip(), top_k=top_k)

        if not hits:
            st.warning("No results found.")
        else:
            # Store hits in session state so they survive button-toggle reruns
            st.session_state["hits"] = hits
            # Reset open state for new search
            st.session_state["open"] = {i: False for i in range(len(hits))}

    except Exception as e:
        st.error(f"Search failed: {e}")

# Render results (persists across reruns from toggle clicks)
if "hits" in st.session_state:
    for i, h in enumerate(st.session_state["hits"]):
        url = h.get("url", "")
        parsed = urlparse(url)
        # Prefer the page title stored at index time; fall back to URL-derived name
        raw_title = h.get("title")
        if raw_title:
            source_name = raw_title if len(raw_title) <= 50 else raw_title[:47] + "..."
        else:
            source_name = parsed.path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").title() if parsed.path.rstrip("/") else parsed.netloc
            if not source_name.strip():
                source_name = parsed.netloc or "Unknown Source"
            if len(source_name) > 50:
                source_name = source_name[:47] + "..."
        clean_path = parsed.scheme + "://" + parsed.netloc + parsed.path
        is_open = st.session_state.get("open", {}).get(i, False)

        # Toggle button — renders as the full formatted header line
        arrow = "▼" if is_open else "▶"
        link = f"[↗ {source_name}]({url})" if url else source_name
        header = f"{arrow} [{i+1}] score={h['score']:.4f} | {link}"

        if st.button(header, key=f"toggle_{i}"):
            st.session_state["open"][i] = not is_open
            st.rerun()

        # Content — only shown when open
        if is_open:
            with st.container():
                if url:
                    st.markdown(f"[↗ {source_name}]({clean_path})")
                st.write("This section discusses:")

                practice = [m["tag"] for m in h.get("practice_tags", [])]
                ethics = [m["tag"] for m in h.get("ethics_tags", [])]
                jurisdictions = [m["tag"] for m in h.get("jurisdiction_tags", [])]

                if practice:
                    st.write(f"• {', '.join(t.capitalize() for t in practice)} practices")
                if ethics:
                    st.write(f"• {', '.join(t.capitalize() for t in ethics)} considerations")
                if jurisdictions:
                    st.write(f"• Relevance to {', '.join(jurisdictions)}")

                st.write("**Excerpt:**")
                raw_text = h.get("text", "")
                if raw_text:
                    display_text = raw_text[:1].upper() + raw_text[1:]
                    if raw_text[-1] not in ".!?":
                        display_text = display_text + "..."
                    st.write(display_text)

            st.divider()