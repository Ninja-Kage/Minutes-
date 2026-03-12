import streamlit as st
import recorder
import transcriber
import rag

st.set_page_config(page_title="Meeting Recorder", page_icon="🎙️", layout="centered")
st.title("🎙️ Meeting Recorder & Transcriber")

# --- Session State ---
for key, default in {
    "recording": False,
    "audio_file": None,
    "transcript": None,
    "current_index": None,
    "chat_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── RECORD / STOP ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("🔴 Record Meeting", disabled=st.session_state.recording, use_container_width=True):
        recorder.start()
        st.session_state.recording = True
        st.session_state.audio_file = None
        st.session_state.transcript = None
        st.session_state.current_index = None
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("⏹️ Stop & Transcribe", disabled=not st.session_state.recording, use_container_width=True):
        recorder.stop()
        st.session_state.audio_file = recorder.save("meeting.wav")
        st.session_state.recording = False
        st.rerun()

if st.session_state.recording:
    st.info("🔴 Recording in progress...")

# ── TRANSCRIPTION ──────────────────────────────────────────────
if st.session_state.audio_file and st.session_state.transcript is None:
    st.audio(st.session_state.audio_file)
    with st.spinner("Transcribing and identifying speakers..."):
        try:
            st.session_state.transcript = transcriber.run(st.session_state.audio_file)
        except Exception as e:
            st.error(f"Transcription error: {e}")
    st.rerun()

# ── TRANSCRIPT DISPLAY ─────────────────────────────────────────
if st.session_state.transcript:
    st.audio(st.session_state.audio_file)
    st.subheader("📝 Transcript")

    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    speakers = sorted(set(s["speaker"] for s in st.session_state.transcript))
    color_map = {spk: COLORS[i % len(COLORS)] for i, spk in enumerate(speakers)}

    for seg in st.session_state.transcript:
        color = color_map[seg["speaker"]]
        st.markdown(
            f"<div style='border-left:4px solid {color}; padding:8px 12px; margin-bottom:10px; background:#f9f9f9;'>"
            f"<b style='color:{color}'>{seg['speaker']}</b> "
            f"<span style='color:#888; font-size:12px'>{seg['start']}s – {seg['end']}s</span><br>"
            f"{seg['text']}</div>",
            unsafe_allow_html=True
        )

    text = "\n\n".join(f"[{s['speaker']} | {s['start']}s-{s['end']}s]\n{s['text']}" for s in st.session_state.transcript)
    st.download_button("⬇️ Download Transcript", text, file_name="transcript.txt")

    st.divider()

    # ── SAVE TO PINECONE ───────────────────────────────────────
    if st.session_state.current_index is None:
        with st.spinner("Saving meeting to Pinecone..."):
            try:
                st.session_state.current_index = rag.store_meeting(st.session_state.transcript)
                st.success(f"✅ Saved as `{st.session_state.current_index}`")
            except Exception as e:
                st.error(f"Pinecone error: {e}")

    # ── Q&A SECTION ────────────────────────────────────────────
    st.subheader("💬 Ask About Meetings")

    # Let user pick which meeting(s) to query
    all_meetings = rag.get_all_meetings()
    query_scope = st.radio(
        "Query scope:",
        ["This meeting only", "All past meetings"],
        horizontal=True
    )

    if query_scope == "This meeting only" and len(all_meetings) > 1:
        selected = st.selectbox("Select a meeting:", all_meetings, index=all_meetings.index(st.session_state.current_index) if st.session_state.current_index in all_meetings else 0)
    else:
        selected = st.session_state.current_index

    # Chat history
    for item in st.session_state.chat_history:
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**Assistant:** {item['answer']}")
        st.markdown("---")

    question = st.text_input("Ask anything...", placeholder="e.g. What tasks were assigned? Who is responsible for X?")

    if st.button("Ask", use_container_width=True) and question:
        with st.spinner("Thinking..."):
            try:
                if query_scope == "All past meetings":
                    answer = rag.query_all_meetings(question)
                else:
                    answer = rag.query_meeting(selected, question)
                st.session_state.chat_history.append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"Query error: {e}")
        st.rerun()

# ── SIDEBAR: PAST MEETINGS ─────────────────────────────────────
with st.sidebar:
    st.header("📁 Past Meetings")
    meetings = rag.get_all_meetings()
    if meetings:
        for m in meetings:
            st.markdown(f"- `{m}`")
    else:
        st.caption("No meetings stored yet.")