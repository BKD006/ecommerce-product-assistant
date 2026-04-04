import requests
import streamlit as st

API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Novacart AI",
    layout="wide",
)

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "ui_session"


# =========================
# API FUNCTIONS
# =========================

def chat_api(query):

    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
    }

    try:
        r = requests.post(
            f"{API_URL}/chat/query",
            json=payload,
        )

        if r.status_code == 200:
            return r.json().get("answer", "No response")

        return f"❌ Error: {r.text}"

    except Exception as e:
        return f"❌ API Error: {str(e)}"


def upload_product(file):

    files = {
        "file": (file.name, file.getvalue(), "text/csv")
    }

    try:
        r = requests.post(
            f"{API_URL}/products/ingest",
            files=files,
        )
        return r.json() if r.status_code == 200 else r.text

    except Exception as e:
        return str(e)


def upload_policy(file):

    files = {
        "file": (file.name, file.getvalue(), "application/pdf")
    }

    try:
        r = requests.post(
            f"{API_URL}/policies/ingest",
            files=files,
        )
        return r.json() if r.status_code == 200 else r.text

    except Exception as e:
        return str(e)


# 🔥 NEW (SQLite ingestion)
def upload_sqlite(file):

    files = {
        "file": (file.name, file.getvalue(), "text/csv")
    }

    try:
        r = requests.post(
            f"{API_URL}/sqlite/ingest",
            files=files,
        )
        return r.json() if r.status_code == 200 else r.text

    except Exception as e:
        return str(e)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🛒 Novacart UI")

    # -------------------------
    # PRODUCT INGESTION
    # -------------------------
    st.subheader("📦 Upload Product CSV (Vector DB)")

    csv_file = st.file_uploader("CSV file", type=["csv"], key="prod")

    if st.button("Ingest Products"):
        if csv_file:
            with st.spinner("Uploading to Pinecone..."):
                res = upload_product(csv_file)
            st.json(res)

    st.divider()

    # -------------------------
    # SQLITE INGESTION
    # -------------------------
    st.subheader("🗄️ Upload Product CSV (SQLite)")

    sqlite_file = st.file_uploader("CSV file", type=["csv"], key="sqlite")

    if st.button("Ingest to SQLite"):
        if sqlite_file:
            with st.spinner("Ingesting into SQLite..."):
                res = upload_sqlite(sqlite_file)
            st.json(res)

    st.divider()

    # -------------------------
    # POLICY INGESTION
    # -------------------------
    st.subheader("📄 Upload Policy PDF")

    pdf_file = st.file_uploader("PDF file", type=["pdf"], key="pdf")

    if st.button("Ingest Policy"):
        if pdf_file:
            with st.spinner("Processing policy..."):
                res = upload_policy(pdf_file)
            st.json(res)


# =========================
# TITLE
# =========================

st.title("🛒 Novacart Assistant")


# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    if msg["role"] == "user":

        col1, col2 = st.columns([3, 7])

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color:#1f2937;
                    padding:10px;
                    border-radius:10px;
                    text-align:right;
                ">
                {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        col1, col2 = st.columns([7, 3])

        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color:#111827;
                    padding:10px;
                    border-radius:10px;
                    text-align:left;
                ">
                {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================
# INPUT
# =========================

if prompt := st.chat_input("Ask something..."):

    # USER MESSAGE
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    col1, col2 = st.columns([3, 7])

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:10px;
                border-radius:10px;
                text-align:right;
            ">
            {prompt}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ASSISTANT RESPONSE
    with st.spinner("Thinking..."):
        answer = chat_api(prompt)

    col1, col2 = st.columns([7, 3])

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:10px;
                border-radius:10px;
                text-align:left;
            ">
            {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )