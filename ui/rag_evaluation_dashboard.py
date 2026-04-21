import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

# ======================================================
# CONFIG
# ======================================================
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag_system"

# ======================================================
# LOAD DATA (UPDATED)
# ======================================================
@st.cache_data(ttl=30)
def load_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # direct evaluations collection
    data = list(db["evaluations"].find({}))

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df


# ======================================================
# UI
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 RAG Evaluation Dashboard")

df = load_data()

if df.empty:
    st.warning("No evaluation data found.")
    st.stop()

# ======================================================
# METRICS (UPDATED)
# ======================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Grounding Score", round(df["grounding_score"].mean(), 3))

with col2:
    st.metric("Avg Semantic Similarity", round(df["semantic_similarity"].mean(), 3))

with col3:
    st.metric("Avg LLM Score", round(df["llm_score"].mean(), 3))

with col4:
    # failure rate
    bad_rate = df["is_bad"].mean() * 100
    st.metric("Bad Response Rate (%)", round(bad_rate, 2))


# ======================================================
# TIME SERIES
# ======================================================
st.subheader("📈 Grounding Score Over Time")

if "timestamp" in df:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    st.line_chart(df.set_index("timestamp")["grounding_score"])


# ======================================================
# FAILURE DISTRIBUTION (NEW)
# ======================================================
st.subheader("🚨 Failure Distribution")

failure_counts = df["is_bad"].value_counts()
st.bar_chart(failure_counts)


# ======================================================
# LLM VERDICT
# ======================================================
if "llm_verdict" in df:
    st.subheader("🧠 LLM Verdict Distribution")
    verdict_counts = df["llm_verdict"].value_counts()
    st.bar_chart(verdict_counts)


# ======================================================
# TOP FAILING QUERIES
# ======================================================
st.subheader("🔥 Top Failing Queries")

top_failures = (
    df[df["is_bad"] == True]
    .groupby("query")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_failures.rename("fail_count"))


# ======================================================
# FILTERS
# ======================================================
st.subheader("🔍 Filter")

min_score = st.slider("Min Grounding Score", 0.0, 1.0, 0.0)
show_only_bad = st.checkbox("Show only bad responses")

filtered_df = df[df["grounding_score"] >= min_score]

if show_only_bad:
    filtered_df = filtered_df[filtered_df["is_bad"] == True]


# ======================================================
# TABLE VIEW
# ======================================================
st.subheader("📋 Detailed Results")

st.dataframe(filtered_df[[
    "query",
    "grounding_score",
    "semantic_similarity",
    "llm_score",
    "is_bad",
    "answer"
]])