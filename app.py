"""Responsive Streamlit UI for the Vietnamese Labour Law RAG assistant.

Run:
    streamlit run app.py

The UI is deliberately usable before Tasks 9-10 are complete: retrieval and
generation failures are converted into clear product states instead of raw
tracebacks.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Trợ lý Pháp luật Lao động",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


NAV_ITEMS = {
    "ask": "Hỏi trợ lý",
    "topics": "Chủ đề pháp luật",
    "history": "Lịch sử tư vấn",
    "saved": "Văn bản đã lưu",
    "guide": "Hướng dẫn sử dụng",
    "support": "Liên hệ hỗ trợ",
}

SUGGESTED_QUESTIONS = [
    "Công ty nợ lương thì tôi phải làm thế nào?",
    "Nghỉ việc có cần báo trước không?",
    "Tôi có được hưởng bảo hiểm thất nghiệp không?",
    "Công ty không đóng bảo hiểm xã hội thì xử lý thế nào?",
    "Làm thêm giờ được tính lương ra sao?",
    "Bị sa thải không đúng quy định thì phải làm gì?",
]

LEGAL_TOPICS = [
    ("Hợp đồng & thử việc", "Giao kết, thử việc, thay đổi và chấm dứt hợp đồng."),
    ("Tiền lương", "Lương tối thiểu, chậm lương, khấu trừ và thưởng."),
    ("Thời giờ làm việc", "Làm thêm giờ, làm ban đêm và thời gian nghỉ."),
    ("Nghỉ việc & sa thải", "Báo trước, trợ cấp và xử lý kỷ luật lao động."),
    ("Bảo hiểm", "Bảo hiểm xã hội, thất nghiệp và quyền lợi liên quan."),
    ("An toàn lao động", "Tai nạn lao động, sức khỏe và môi trường làm việc."),
]

LEGAL_NOTICE = (
    "Thông tin được cung cấp nhằm mục đích tham khảo và không thay thế ý kiến "
    "tư vấn của luật sư hoặc cơ quan có thẩm quyền."
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "active_view": "ask",
        "messages": [],
        "pending_query": None,
        "saved_sources": [],
        "saved_answers": [],
        "font_scale": "Mặc định",
        "high_contrast": False,
        "top_k": 5,
        "feedback": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_theme() -> None:
    font_sizes = {"Mặc định": "17px", "Chữ lớn": "19px"}
    body_size = font_sizes[st.session_state.font_scale]
    if st.session_state.high_contrast:
        palette = {
            "canvas": "#FFFFFF",
            "surface": "#FFFFFF",
            "navy": "#0A2342",
            "slate": "#244E73",
            "text": "#101820",
            "muted": "#394553",
            "border": "#8E9AA5",
        }
    else:
        palette = {
            "canvas": "#F8F7F3",
            "surface": "#FFFFFF",
            "navy": "#183153",
            "slate": "#31577A",
            "text": "#202A35",
            "muted": "#66717D",
            "border": "#DDE2E6",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --canvas: {palette['canvas']};
            --surface: {palette['surface']};
            --navy: {palette['navy']};
            --slate: {palette['slate']};
            --text: {palette['text']};
            --muted: {palette['muted']};
            --border: {palette['border']};
            --champagne: #B89455;
            --positive: #28745A;
            --warning: #B86532;
            --danger: #A64040;
        }}

        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: "Be Vietnam Pro", "Segoe UI", Arial, sans-serif;
            color: var(--text);
        }}
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{ color: inherit; }}
        html {{ font-size: {body_size}; }}
        .stApp {{ background: var(--canvas); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ background: transparent; }}
        [data-testid="stAppDeployButton"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .block-container {{
            max-width: 1480px;
            padding-top: 1.8rem;
            padding-bottom: 5.5rem;
        }}

        [data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] .block-container {{ padding-top: 1.35rem; }}
        [data-testid="stSidebar"] button {{ text-align: left; }}

        .brand {{
            display: grid;
            grid-template-columns: 44px 1fr;
            gap: 12px;
            align-items: center;
            padding: 4px 0 18px;
        }}
        .brand-mark {{
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: var(--navy);
            color: var(--champagne);
            font-weight: 800;
            letter-spacing: .02em;
        }}
        .brand-name {{ color: var(--navy); font-size: 1rem; font-weight: 800; line-height: 1.25; }}
        .brand-caption {{ color: var(--muted); font-size: .76rem; margin-top: 3px; }}

        .eyebrow {{
            color: var(--slate);
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }}
        .hero {{ padding: 2.1rem 0 1.25rem; max-width: 850px; }}
        .hero h1 {{
            color: var(--navy);
            font-size: clamp(2rem, 4vw, 3.65rem);
            line-height: 1.08;
            letter-spacing: -.025em;
            margin: 0 0 1rem;
        }}
        .hero p {{ color: var(--muted); font-size: 1.05rem; line-height: 1.75; max-width: 760px; }}

        .section-title {{
            color: var(--navy);
            font-size: 1.25rem;
            font-weight: 800;
            margin: 1rem 0 .7rem;
        }}
        .muted {{ color: var(--muted); line-height: 1.65; }}

        .status-line {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 34px;
            padding: 6px 11px;
            border: 1px solid #BFD8CD;
            border-radius: 999px;
            color: var(--positive);
            background: #F2F8F5;
            font-size: .82rem;
            font-weight: 700;
        }}
        .status-line::before {{ content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }}

        .source-panel, .info-panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 10px 34px rgba(24, 49, 83, .055);
            padding: 1.15rem;
        }}
        .source-panel {{ position: sticky; top: 1rem; }}
        .source-empty {{
            border: 1px dashed var(--border);
            border-radius: 12px;
            padding: 1.1rem;
            color: var(--muted);
            line-height: 1.55;
            background: #FBFBF9;
        }}

        .notice {{
            border-left: 3px solid var(--champagne);
            background: #FBF8F1;
            color: #53606C;
            border-radius: 0 10px 10px 0;
            padding: .8rem 1rem;
            font-size: .82rem;
            line-height: 1.55;
            margin-top: 1rem;
        }}

        [data-testid="stChatMessage"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: .35rem .65rem;
            box-shadow: 0 6px 20px rgba(24, 49, 83, .035);
        }}
        [data-testid="stChatMessage"] {{ color: var(--text); }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
            color: var(--text) !important;
        }}
        [data-testid="stCaptionContainer"] p {{ color: var(--muted) !important; }}
        [data-testid="stChatInput"] {{ border-color: var(--border); }}

        .stButton > button, .stFormSubmitButton > button, .stLinkButton > a {{
            min-height: 44px;
            border-radius: 10px;
            border-color: var(--border);
            font-weight: 700;
            white-space: normal;
        }}
        .stButton > button[data-testid="stBaseButton-primary"],
        .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"] {{
            background: var(--navy);
            border-color: var(--navy);
            color: #FFFFFF;
        }}
        .stButton > button[data-testid="stBaseButton-secondary"] {{
            background: var(--surface);
            border-color: var(--border);
            color: var(--navy);
        }}
        .stTextArea textarea {{
            min-height: 132px;
            border-radius: 14px;
            border-color: var(--border);
            font-size: 1rem;
            line-height: 1.55;
            background: var(--surface);
            color: var(--text);
        }}
        .stTextArea textarea::placeholder {{ color: var(--muted); opacity: 1; }}
        .stTextInput input {{
            min-height: 46px;
            border-radius: 10px;
            border-color: var(--border);
            background: var(--surface);
            color: var(--text);
        }}
        .stTextInput input::placeholder {{ color: var(--muted); opacity: 1; }}

        @media (max-width: 900px) {{
            .block-container {{ padding: 1rem 1rem 5rem; }}
            .hero {{ padding-top: 1.1rem; }}
            .hero h1 {{ font-size: 2.25rem; }}
            .source-panel {{ position: static; margin-top: 1rem; }}
            .stButton > button {{ width: 100%; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ scroll-behavior: auto !important; transition: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def go_to(view: str) -> None:
    st.session_state.active_view = view


def queue_question(question: str) -> None:
    st.session_state.pending_query = question
    st.session_state.active_view = "ask"


def source_identity(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    return str(
        meta.get("document_title")
        or meta.get("title")
        or meta.get("source")
        or "Nguồn chưa xác định"
    )


def source_citation(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    parts = [meta.get("article"), meta.get("document_number") or meta.get("year")]
    return " · ".join(str(part) for part in parts if part) or "Chưa có thông tin điều/khoản"


def latest_sources() -> list[dict[str, Any]]:
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant" and message.get("sources"):
            return message["sources"]
    return []


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">PL</div>
              <div>
                <div class="brand-name">Trợ lý Pháp luật<br>Lao động</div>
                <div class="brand-caption">Hiểu luật để bảo vệ quyền lợi</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for key, label in NAV_ITEMS.items():
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if st.session_state.active_view == key else "secondary",
            ):
                go_to(key)
                st.rerun()

        st.divider()
        st.markdown("**Cấu hình tra cứu**")
        st.slider(
            "Số nguồn truy xuất",
            min_value=3,
            max_value=8,
            key="top_k",
            help="Số đoạn tài liệu tối đa được đưa vào câu trả lời.",
        )

        st.divider()
        st.markdown("**Khả năng tiếp cận**")
        st.selectbox(
            "Cỡ chữ",
            ["Mặc định", "Chữ lớn"],
            key="font_scale",
            label_visibility="collapsed",
        )
        st.toggle("Tương phản cao", key="high_contrast")
        st.markdown(f'<div class="notice">{LEGAL_NOTICE}</div>', unsafe_allow_html=True)


def save_source(source: dict[str, Any]) -> None:
    identity = source_identity(source)
    if not any(source_identity(item) == identity for item in st.session_state.saved_sources):
        st.session_state.saved_sources.append(source)
        st.toast("Đã lưu văn bản", icon="✅")
    else:
        st.toast("Văn bản đã có trong danh sách lưu")


def render_source_panel(sources: list[dict[str, Any]]) -> None:
    st.markdown('<div class="source-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Nguồn đã truy xuất</div>', unsafe_allow_html=True)
    if not sources:
        st.markdown(
            """
            <div class="source-empty">
              Nguồn pháp lý sẽ xuất hiện tại đây sau khi trợ lý phân tích câu hỏi.
              Hệ thống không hiển thị phần trăm chính xác khi chưa có phương pháp đo đáng tin cậy.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-line">Có căn cứ liên quan</span>', unsafe_allow_html=True)
        for index, source in enumerate(sources, 1):
            meta = source.get("metadata") or {}
            with st.expander(f"{index}. {source_identity(source)}", expanded=index == 1):
                st.caption(source_citation(source))
                score = source.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    st.caption(f"Điểm truy xuất: {score:.4f}")
                content = str(source.get("content") or "")
                st.write(content[:650] + ("…" if len(content) > 650 else ""))
                issued = meta.get("effective_date") or meta.get("date") or meta.get("year")
                if issued:
                    st.caption(f"Ngày ban hành/hiệu lực: {issued}")
                url = meta.get("url") or meta.get("source_url")
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    st.link_button("Mở nội dung gốc", url, width="stretch")
                if st.button("Lưu văn bản", key=f"save_source_{index}", width="stretch"):
                    save_source(source)
    st.markdown("</div>", unsafe_allow_html=True)


def render_answer_actions(index: int, message: dict[str, Any]) -> None:
    cols = st.columns([1, 1, 1.15])
    with cols[0]:
        if st.button("Lưu câu trả lời", key=f"save_answer_{index}", width="stretch"):
            if message["content"] not in st.session_state.saved_answers:
                st.session_state.saved_answers.append(message["content"])
            st.toast("Đã lưu câu trả lời", icon="✅")
    with cols[1]:
        if st.button("Chưa phù hợp", key=f"feedback_{index}", width="stretch"):
            st.session_state.feedback[str(index)] = "not_helpful"
            st.toast("Đã ghi nhận phản hồi")
    with cols[2]:
        if st.button("Cần chuyên viên", key=f"expert_{index}", width="stretch"):
            go_to("support")
            st.rerun()


def process_query(query: str) -> None:
    clean_query = query.strip()
    if not clean_query:
        return

    st.session_state.messages.append(
        {"role": "user", "content": clean_query, "created_at": datetime.now().isoformat()}
    )

    with st.status("Đang tìm kiếm văn bản pháp luật…", expanded=True) as status:
        try:
            st.write("Đang phân tích tình huống và đối chiếu nguồn liên quan.")
            from src.task10_generation import generate_with_citation

            response = generate_with_citation(
                clean_query,
                top_k=int(st.session_state.get("top_k", 5)),
            )
            answer = response.get("answer") or "Chưa thể tạo câu trả lời từ nguồn hiện có."
            sources = response.get("sources") or []
            retrieval_source = response.get("retrieval_source", "none")
            status.update(label="Đã hoàn tất đối chiếu nguồn", state="complete", expanded=False)
            message_state = "evidence" if sources else "insufficient"
        except NotImplementedError:
            answer = (
                "Giao diện đã sẵn sàng nhưng mô-đun sinh câu trả lời đang được nhóm hoàn thiện. "
                "Bạn có thể tiếp tục khám phá các màn hình và câu hỏi gợi ý."
            )
            sources = []
            retrieval_source = "none"
            message_state = "system"
            status.update(label="Task 10 chưa được kết nối", state="error", expanded=False)
        except Exception as exc:  # UI boundary: never expose raw traceback or secrets.
            error_name = type(exc).__name__
            answer = (
                "Hệ thống chưa thể xử lý câu hỏi lúc này. Vui lòng kiểm tra cấu hình dữ liệu/API "
                "hoặc thử lại sau."
            )
            sources = []
            retrieval_source = "none"
            message_state = "error"
            status.update(label=f"Không thể hoàn tất ({error_name})", state="error", expanded=False)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "state": message_state,
            "created_at": datetime.now().isoformat(),
        }
    )


def render_chat_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                state = message.get("state")
                if state == "evidence":
                    st.success("Có căn cứ liên quan", icon="✅")
                elif state in {"insufficient", "error"}:
                    st.warning("Chưa đủ thông tin để kết luận", icon="⚠️")
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(LEGAL_NOTICE)
                render_answer_actions(index, message)


def render_ask_view() -> None:
    sources = latest_sources()
    main_col, source_col = st.columns([1.72, 0.78], gap="large")

    with main_col:
        if not st.session_state.messages:
            st.markdown(
                """
                <section class="hero">
                  <div class="eyebrow">Tra cứu dựa trên nguồn pháp luật</div>
                  <h1>Bạn đang gặp vấn đề gì trong công việc?</h1>
                  <p>Hãy mô tả tình huống bằng ngôn ngữ thông thường. Trợ lý sẽ phân tích,
                  giải thích quy định liên quan và cung cấp nguồn tham khảo.</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<span class="status-line">Sẵn sàng tra cứu</span>', unsafe_allow_html=True)

            with st.form("hero_question_form", clear_on_submit=True):
                initial_query = st.text_area(
                    "Câu hỏi của bạn",
                    placeholder=(
                        "Ví dụ: Công ty cho tôi nghỉ việc nhưng không báo trước thì tôi có "
                        "được bồi thường không?"
                    ),
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "Phân tích tình huống", type="primary", width="stretch"
                )
            if submitted and initial_query.strip():
                process_query(initial_query)
                st.rerun()

            st.markdown('<div class="section-title">Câu hỏi thường gặp</div>', unsafe_allow_html=True)
            question_columns = st.columns(2)
            for index, question in enumerate(SUGGESTED_QUESTIONS):
                with question_columns[index % 2]:
                    if st.button(question, key=f"suggestion_{index}", width="stretch"):
                        queue_question(question)
                        st.rerun()
        else:
            st.markdown('<div class="eyebrow">Phiên tư vấn hiện tại</div>', unsafe_allow_html=True)
            st.markdown("## Trao đổi với trợ lý")
            render_chat_history()

            with st.form("follow_up_form", clear_on_submit=True):
                query = st.text_input(
                    "Câu hỏi tiếp theo",
                    placeholder="Mô tả thêm tình huống hoặc đặt câu hỏi tiếp theo…",
                    label_visibility="collapsed",
                )
                follow_up_submitted = st.form_submit_button(
                    "Gửi câu hỏi", type="primary", width="stretch"
                )
            if follow_up_submitted and query.strip():
                process_query(query)
                st.rerun()

        pending = st.session_state.pop("pending_query", None)
        if pending:
            process_query(pending)
            st.rerun()

    with source_col:
        render_source_panel(sources)


def render_topics_view() -> None:
    st.markdown('<div class="eyebrow">Thư viện chủ đề</div>', unsafe_allow_html=True)
    st.title("Chủ đề pháp luật lao động")
    st.markdown(
        '<p class="muted">Chọn một chủ đề để bắt đầu bằng tình huống gần với vấn đề của bạn.</p>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, (title, description) in enumerate(LEGAL_TOPICS):
        with columns[index % 2]:
            with st.container(border=True):
                st.subheader(title)
                st.write(description)
                if st.button("Đặt câu hỏi về chủ đề này", key=f"topic_{index}"):
                    queue_question(f"Tôi cần được hướng dẫn về {title.lower()}.")
                    st.rerun()


def render_history_view() -> None:
    st.markdown('<div class="eyebrow">Lịch sử tư vấn</div>', unsafe_allow_html=True)
    st.title("Các trao đổi trong phiên này")
    if not st.session_state.messages:
        st.info("Bạn chưa có cuộc trao đổi nào trong phiên hiện tại.")
        return
    for index, message in enumerate(st.session_state.messages):
        role = "Bạn" if message["role"] == "user" else "Trợ lý"
        with st.expander(f"{role} · {index + 1}"):
            st.markdown(message["content"])
    if st.button("Xóa lịch sử phiên", type="secondary"):
        st.session_state.messages = []
        st.rerun()


def render_saved_view() -> None:
    st.markdown('<div class="eyebrow">Tủ tài liệu cá nhân</div>', unsafe_allow_html=True)
    st.title("Nội dung đã lưu")
    if not st.session_state.saved_sources and not st.session_state.saved_answers:
        st.info("Chưa có văn bản hoặc câu trả lời nào được lưu trong phiên này.")
        return
    if st.session_state.saved_sources:
        st.subheader("Văn bản pháp luật")
        for source in st.session_state.saved_sources:
            with st.expander(source_identity(source)):
                st.caption(source_citation(source))
                st.write(str(source.get("content") or "")[:800])
    if st.session_state.saved_answers:
        st.subheader("Câu trả lời")
        for index, answer in enumerate(st.session_state.saved_answers, 1):
            with st.expander(f"Câu trả lời đã lưu {index}"):
                st.markdown(answer)


def render_guide_view() -> None:
    st.markdown('<div class="eyebrow">Bắt đầu nhanh</div>', unsafe_allow_html=True)
    st.title("Hướng dẫn sử dụng")
    steps = [
        ("1. Mô tả sự việc", "Nêu thời điểm, loại hợp đồng và điều công ty đã thực hiện."),
        ("2. Không nhập dữ liệu nhạy cảm", "Che số CCCD, tài khoản, địa chỉ và thông tin sức khỏe."),
        ("3. Đọc nguồn đi kèm", "Mở văn bản gốc và kiểm tra điều, khoản được trích dẫn."),
        ("4. Tìm hỗ trợ khi cần", "Trường hợp rủi ro cao nên gặp luật sư hoặc cơ quan có thẩm quyền."),
    ]
    for title, description in steps:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.warning(LEGAL_NOTICE)


def render_support_view() -> None:
    st.markdown('<div class="eyebrow">Hỗ trợ bổ sung</div>', unsafe_allow_html=True)
    st.title("Yêu cầu chuyên viên xem xét")
    st.markdown(
        "Tính năng kết nối chuyên viên chưa được tích hợp. Bạn có thể chuẩn bị trước phần tóm tắt "
        "tình huống và danh sách tài liệu đang có."
    )
    with st.form("support_draft"):
        st.text_input("Chủ đề", placeholder="Ví dụ: Chấm dứt hợp đồng không báo trước")
        st.text_area(
            "Tóm tắt tình huống",
            placeholder="Không nhập CCCD, số tài khoản hoặc thông tin cá nhân nhạy cảm.",
        )
        acknowledged = st.checkbox("Tôi hiểu biểu mẫu này chưa gửi dữ liệu ra ngoài hệ thống")
        if st.form_submit_button("Lưu bản nháp trong phiên", width="stretch"):
            if acknowledged:
                st.success("Đã ghi nhận bản nháp trong phiên trình duyệt.")
            else:
                st.warning("Vui lòng xác nhận trước khi lưu bản nháp.")


def render_active_view() -> None:
    renderers = {
        "ask": render_ask_view,
        "topics": render_topics_view,
        "history": render_history_view,
        "saved": render_saved_view,
        "guide": render_guide_view,
        "support": render_support_view,
    }
    renderers.get(st.session_state.active_view, render_ask_view)()


init_state()
inject_theme()
render_sidebar()
render_active_view()
