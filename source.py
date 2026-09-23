"""Public Streamlit interface for PRRISM."""

import html
import re

import streamlit as st
import streamlit.components.v1 as components

from answer_generator import (
    build_evidence,
    generate_answer,
)
from search_engine import (
    load_artifacts,
    search,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="PRRISM",
    page_icon="◌",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# STYLES
# =============================================================================

st.markdown(
    """
<style>

:root {
    --background: #fbfbfd;
    --surface: rgba(255, 255, 255, 0.94);

    --text: #1d1d1f;
    --secondary: #6e6e73;
    --tertiary: #86868b;

    --border: rgba(0, 0, 0, 0.09);
    --soft: #f1f1f3;
}


/* --------------------------------------------------------------------------
   Base
--------------------------------------------------------------------------- */

html,
body,
[class*="css"] {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Helvetica Neue",
        Arial,
        sans-serif;
}

html,
body,
.stApp {
    background: var(--background);
}

.stApp {
    color: var(--text);
}

.block-container {
    max-width: 900px;
    padding-top: 1.8rem;
    padding-bottom: 10rem;
}


/* --------------------------------------------------------------------------
   Hide Streamlit chrome
--------------------------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}


/* --------------------------------------------------------------------------
   Header
--------------------------------------------------------------------------- */

.prrism-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    width: 100%;
    margin: 0 auto;
}

.prrism-wordmark {
    color: var(--text);

    font-size: 0.92rem;
    font-weight: 700;

    letter-spacing: 0.16em;
}

.prrism-nav {
    color: var(--secondary);

    font-size: 0.8rem;
    font-weight: 450;

    letter-spacing: -0.01em;
}


/* --------------------------------------------------------------------------
   Landing hero
--------------------------------------------------------------------------- */

.prrism-hero {
    max-width: 760px;

    margin:
        clamp(7rem, 17vh, 11rem)
        auto
        3rem;

    text-align: center;

    animation:
        hero-enter
        650ms
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}

@keyframes hero-enter {

    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

.prrism-title {
    margin: 0;

    color: var(--text);

    font-size:
        clamp(
            3.7rem,
            7vw,
            5.5rem
        );

    font-weight: 600;
    line-height: 0.98;

    letter-spacing: -0.06em;
}

.prrism-subtitle {
    max-width: 570px;

    margin:
        1.5rem
        auto
        0;

    color: var(--secondary);

    font-size: 1.3rem;
    font-weight: 400;
    line-height: 1.42;

    letter-spacing: -0.025em;
}

.prrism-corpus-note {
    margin-top: 1.7rem;

    color: var(--tertiary);

    font-size: 0.73rem;

    text-align: center;

    letter-spacing: -0.005em;
}


/* --------------------------------------------------------------------------
   Conversation
--------------------------------------------------------------------------- */

.conversation {
    margin-top: 5.5rem;
}


/* --------------------------------------------------------------------------
   User question
--------------------------------------------------------------------------- */

.user-question-wrap {
    max-width: 760px;

    margin:
        2rem
        0
        3.8rem;
}

.user-question-label {
    margin-bottom: 0.65rem;

    color: var(--tertiary);

    font-size: 0.72rem;
    font-weight: 600;
}

.user-question {
    color: var(--text);

    font-size: 1.65rem;
    font-weight: 500;
    line-height: 1.3;

    letter-spacing: -0.035em;
}


/* --------------------------------------------------------------------------
   PRRISM ranked results
--------------------------------------------------------------------------- */

.results-wrap {
    max-width: 760px;

    margin-bottom: 1rem;
}

.results-header {
    display: flex;

    align-items: baseline;
    justify-content: space-between;

    gap: 1rem;

    margin-bottom: 0.8rem;
}

.results-label {
    color: var(--tertiary);

    font-size: 0.72rem;
    font-weight: 600;

    letter-spacing: 0.01em;
}

.results-context {
    color: var(--tertiary);

    font-size: 0.68rem;
    font-weight: 400;

    letter-spacing: -0.005em;
}


/* --------------------------------------------------------------------------
   Individual ranked result
--------------------------------------------------------------------------- */

.ranked-result {
    display: grid;

    grid-template-columns:
        2.7rem
        minmax(0, 1fr);

    column-gap: 1.1rem;

    padding:
        1.75rem
        0
        1.9rem;

    border-top:
        1px solid
        var(--border);
}

.ranked-result:last-child {
    border-bottom:
        1px solid
        var(--border);
}

.result-rank {
    padding-top: 0.08rem;

    color: var(--tertiary);

    font-size: 0.72rem;
    font-weight: 500;

    letter-spacing: 0.04em;
}

.result-content {
    min-width: 0;
}

.result-title {
    display: inline;

    color: var(--text) !important;

    font-size: 1rem;
    font-weight: 600;
    line-height: 1.4;

    letter-spacing: -0.018em;

    text-decoration: none !important;
}

.result-title:hover {
    opacity: 0.65;
}

.result-domain {
    display: inline;

    margin-left: 0.45rem;

    color: var(--tertiary);

    font-size: 0.68rem;
    font-weight: 400;

    white-space: nowrap;
}

.result-summary {
    margin-top: 0.75rem;

    color: var(--text);

    font-size: 1rem;
    font-weight: 400;
    line-height: 1.68;

    letter-spacing: -0.012em;
}

.result-summary p {
    margin:
        0
        0
        0.85rem;
}

.result-summary p:last-child {
    margin-bottom: 0;
}

.result-link-row {
    margin-top: 0.85rem;
}

.result-link {
    display: inline-flex;

    align-items: center;

    gap: 0.3rem;

    color: var(--secondary) !important;

    font-size: 0.7rem;
    font-weight: 500;

    text-decoration: none !important;

    transition:
        color 140ms ease,
        opacity 140ms ease;
}

.result-link:hover {
    color: var(--text) !important;
}


/* --------------------------------------------------------------------------
   Source-count control
--------------------------------------------------------------------------- */

.source-control {
    max-width: 760px;

    margin:
        1.25rem
        auto
        0.25rem;
}

.source-control-label {
    margin-bottom: 0.15rem;

    color: var(--secondary);

    font-size: 0.72rem;
    font-weight: 500;

    letter-spacing: -0.005em;
}

.source-control-note {
    margin-bottom: 0.25rem;

    color: var(--tertiary);

    font-size: 0.66rem;

    letter-spacing: -0.005em;
}

[data-testid="stSlider"] {
    max-width: 760px;

    margin: 0 auto;
}

[data-testid="stSlider"] label {
    display: none;
}

[data-testid="stSlider"] p {
    color: var(--secondary);

    font-size: 0.72rem;
}


/* --------------------------------------------------------------------------
   Empty evidence
--------------------------------------------------------------------------- */

.empty-results {
    max-width: 760px;

    padding:
        1.7rem
        0;

    border-top:
        1px solid
        var(--border);

    border-bottom:
        1px solid
        var(--border);

    color: var(--secondary);

    font-size: 1rem;
    line-height: 1.65;

    letter-spacing: -0.012em;
}


/* --------------------------------------------------------------------------
   Chat input
--------------------------------------------------------------------------- */

[data-testid="stBottom"] {
    background:
        linear-gradient(
            to bottom,
            rgba(251, 251, 253, 0),
            rgba(251, 251, 253, 0.96) 38%,
            #fbfbfd 72%
        ) !important;

    padding-bottom: 1.7rem;
}


[data-testid="stChatInput"] {
    max-width: 760px;

    margin: 0 auto;

    border:
        1px solid
        rgba(0, 0, 0, 0.10) !important;

    border-radius:
        18px !important;

    background:
        rgba(
            255,
            255,
            255,
            0.96
        ) !important;

    box-shadow:
        0 8px 30px
        rgba(0, 0, 0, 0.06);

    backdrop-filter:
        blur(20px);

    -webkit-backdrop-filter:
        blur(20px);

    overflow: hidden;

    transition:
        border-color 160ms ease,
        box-shadow 160ms ease,
        transform 160ms ease;
}


[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div {
    background:
        transparent !important;

    background-color:
        transparent !important;
}


[data-testid="stChatInput"] textarea {
    background:
        transparent !important;

    background-color:
        transparent !important;

    color:
        #1d1d1f !important;

    caret-color:
        #1d1d1f !important;

    -webkit-text-fill-color:
        #1d1d1f !important;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Text",
        "Helvetica Neue",
        Arial,
        sans-serif !important;

    font-size:
        0.96rem !important;

    font-weight:
        400 !important;

    letter-spacing:
        -0.012em !important;

    box-shadow:
        none !important;
}


[data-testid="stChatInput"] textarea::placeholder {
    color:
        #9a9a9f !important;

    -webkit-text-fill-color:
        #9a9a9f !important;

    opacity:
        1 !important;
}


[data-testid="stChatInput"]:focus-within {
    border-color:
        rgba(
            0,
            0,
            0,
            0.18
        ) !important;

    box-shadow:
        0 10px 36px
        rgba(0, 0, 0, 0.09);

    transform:
        translateY(-1px);
}


[data-testid="stChatInputSubmitButton"] {
    border-radius:
        50% !important;

    background:
        #1d1d1f !important;

    color:
        #ffffff !important;

    transition:
        opacity 140ms ease,
        transform 140ms ease;
}

[data-testid="stChatInputSubmitButton"]:hover {
    opacity: 0.78;

    transform:
        translateY(-1px);
}


/* --------------------------------------------------------------------------
   Spinner
--------------------------------------------------------------------------- */

[data-testid="stSpinner"] {
    max-width: 760px;

    margin:
        2rem
        auto;

    color: var(--secondary);
}


/* --------------------------------------------------------------------------
   Errors
--------------------------------------------------------------------------- */

[data-testid="stAlert"] {
    max-width: 760px;

    margin:
        1.5rem
        auto;

    border-radius: 14px;

    font-size: 0.88rem;
}


/* --------------------------------------------------------------------------
   About
--------------------------------------------------------------------------- */

[data-testid="stExpander"] {
    max-width: 760px;

    margin:
        2.2rem
        0
        0;

    border: none !important;

    border-top:
        1px solid
        var(--border) !important;

    border-radius:
        0 !important;

    background:
        transparent !important;
}

[data-testid="stExpander"] summary {
    color: var(--secondary);

    font-size: 0.78rem;
}


/* --------------------------------------------------------------------------
   Mobile
--------------------------------------------------------------------------- */

@media (max-width: 700px) {

    .block-container {
        padding:
            1.25rem
            1.2rem
            9rem;
    }

    .prrism-hero {
        margin-top: 6rem;
    }

    .prrism-title {
        font-size: 3.5rem;
    }

    .prrism-subtitle {
        max-width: 390px;

        font-size: 1.12rem;
    }

    .conversation {
        margin-top: 3.7rem;
    }

    .user-question {
        font-size: 1.42rem;
    }

    .ranked-result {
        grid-template-columns:
            2rem
            minmax(0, 1fr);

        column-gap: 0.75rem;
    }

    .result-summary {
        font-size: 0.98rem;

        line-height: 1.65;
    }

    .result-domain {
        display: block;

        margin:
            0.25rem
            0
            0;
    }

    .results-context {
        display: none;
    }

    .prrism-nav {
        display: none;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# RESOURCES
# =============================================================================

@st.cache_resource
def get_artifacts():
    """Load the active PRRISM retrieval artifacts once."""

    return load_artifacts()


try:

    index, chunks = (
        get_artifacts()
    )

except Exception:

    st.error(
        "PRRISM's research corpus "
        "could not be loaded. "
        "Please try again later."
    )

    st.stop()


# =============================================================================
# HELPERS
# =============================================================================

def source_domain(
    url: str | None,
) -> str:
    """Extract a clean domain from a source URL."""

    if not url:
        return ""

    match = re.match(
        r"https?://(?:www\\.)?([^/]+)",
        url,
    )

    if not match:
        return ""

    return match.group(1)


def display_rank(
    rank: int,
) -> str:
    """Format public result ranks consistently."""

    return f"{rank:02d}"


def render_summary(
    summary: str,
) -> str:
    """Render a generated summary while preserving paragraph breaks."""

    paragraphs = [
        paragraph.strip()
        for paragraph in summary.split("\n")
        if paragraph.strip()
    ]

    return "".join(
        (
            "<p>"
            f"{html.escape(paragraph)}"
            "</p>"
        )
        for paragraph in paragraphs
    )


def render_ranked_results(
    results: list[dict],
) -> None:
    """Render PRRISM's ranked research results."""

    if not results:

        st.markdown(
            (
                '<div class="results-wrap">'
                '<div class="results-label">'
                'PRRISM'
                '</div>'
                '<div class="empty-results">'
                'The PRRISM corpus did not return '
                'relevant evidence for this question.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        return

    result_blocks = []

    for result in results:

        rank = result[
            "rank"
        ]

        title = html.escape(
            result.get(
                "title"
            )
            or "Untitled source"
        )

        summary_html = render_summary(
            result.get(
                "summary"
            )
            or ""
        )

        url = result.get(
            "url"
        )

        domain = source_domain(
            url
        )

        if url:

            escaped_url = (
                html.escape(
                    url,
                    quote=True,
                )
            )

            title_html = (
                f'<a class="result-title" '
                f'href="{escaped_url}" '
                f'target="_blank" '
                f'rel="noopener noreferrer">'
                f'{title}'
                f'</a>'
            )

            link_html = (
                '<div class="result-link-row">'
                f'<a class="result-link" '
                f'href="{escaped_url}" '
                f'target="_blank" '
                f'rel="noopener noreferrer">'
                'View source '
                '<span>↗</span>'
                '</a>'
                '</div>'
            )

        else:

            title_html = (
                '<span class="result-title">'
                f'{title}'
                '</span>'
            )

            link_html = ""

        domain_html = ""

        if domain:

            domain_html = (
                '<span class="result-domain">'
                f'{html.escape(domain)}'
                '</span>'
            )

        result_blocks.append(
            (
                '<article class="ranked-result">'

                '<div class="result-rank">'
                f'{display_rank(rank)}'
                '</div>'

                '<div class="result-content">'

                '<div>'
                f'{title_html}'
                f'{domain_html}'
                '</div>'

                '<div class="result-summary">'
                f'{summary_html}'
                '</div>'

                f'{link_html}'

                '</div>'

                '</article>'
            )
        )

    st.markdown(
        (
            '<div class="results-wrap">'

            '<div class="results-header">'

            '<div class="results-label">'
            'PRRISM'
            '</div>'

            '<div class="results-context">'
            'Ranked by relevance'
            '</div>'

            '</div>'

            f'{"".join(result_blocks)}'

            '</div>'
        ),
        unsafe_allow_html=True,
    )


def answer_question(
    question: str,
    source_count: int,
) -> dict:
    """Run the ranked PRRISM research pipeline.

    The researcher chooses how many unique sources to review.

    Search establishes the ranking and ensures the requested number
    of unique sources is represented when enough candidates exist.

    build_evidence() groups passages by source while preserving
    PRRISM's ranked source order.

    generate_answer() explains each ranked source independently.
    """

    search_results = search(
        index,
        chunks,
        question,
        top_k=source_count,
        source_count=source_count,
    )

    if not search_results:
        return {
            "results": []
        }

    evidence = build_evidence(
        search_results
    )

    return generate_answer(
        question,
        evidence,
    )

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =============================================================================
# HEADER
# =============================================================================

header_action = (
    "New conversation"
    if st.session_state.messages
    else "Research assistant"
)

st.markdown(
    (
        '<div class="prrism-header">'

        '<div class="prrism-wordmark">'
        'PRRISM'
        '</div>'

        '<div class="prrism-nav">'
        f'{header_action}'
        '</div>'

        '</div>'
    ),
    unsafe_allow_html=True,
)


# =============================================================================
# LANDING STATE
# =============================================================================

if not st.session_state.messages:

    st.markdown(
        (
            '<div class="prrism-hero">'

            '<div class="prrism-title">'
            'Ask PRRISM.'
            '</div>'

            '<div class="prrism-subtitle">'
            'Navigate the research behind '
            'facial recognition.'
            '</div>'

            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="prrism-corpus-note">'
            'Built from a curated research '
            '&amp; policy corpus'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# CONVERSATION
# =============================================================================

if st.session_state.messages:

    st.markdown(
        '<div class="conversation">',
        unsafe_allow_html=True,
    )

    for message in (
        st.session_state.messages
    ):

        if (
            message["role"]
            == "user"
        ):

            st.markdown(
                (
                    '<div class="user-question-wrap">'

                    '<div class="user-question-label">'
                    'You'
                    '</div>'

                    '<div class="user-question">'
                    f'{html.escape(message["content"])}'
                    '</div>'

                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        else:

            render_ranked_results(
                message[
                    "results"
                ]
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SOURCE COUNT
# =============================================================================

st.markdown(
    (
        '<div class="source-control">'
        '<div class="source-control-label">'
        'Number of sources'
        '</div>'
        '<div class="source-control-note">'
        'Choose how many ranked sources PRRISM should review'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)

source_count = st.slider(
    "Number of sources",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    label_visibility="collapsed",
)


# =============================================================================
# CHAT INPUT
# =============================================================================

placeholder = (
    ""
    if not st.session_state.messages
    else "Ask a follow-up"
)

question = st.chat_input(
    placeholder
)


# =============================================================================
# ANIMATED LANDING PLACEHOLDER
# =============================================================================

if not st.session_state.messages:

    components.html(
        """
<script>

const prompts = [
    "What are the privacy risks of facial recognition?",
    "How should facial recognition be tested for fairness?",
    "What role should human oversight play?",
    "How does accuracy vary across demographic groups?"
];

let promptIndex = 0;
let characterIndex = 0;
let deleting = false;


function findInput() {

    return window.parent.document.querySelector(
        '[data-testid="stChatInput"] textarea'
    );

}


function animatePlaceholder() {

    const input = findInput();

    if (!input) {

        setTimeout(
            animatePlaceholder,
            200
        );

        return;

    }


    if (input.value.length > 0) {

        setTimeout(
            animatePlaceholder,
            300
        );

        return;

    }


    const prompt =
        prompts[promptIndex];


    if (!deleting) {

        characterIndex += 1;

        input.placeholder =
            prompt.slice(
                0,
                characterIndex
            );


        if (
            characterIndex
            >= prompt.length
        ) {

            deleting = true;

            setTimeout(
                animatePlaceholder,
                2200
            );

            return;

        }


        setTimeout(
            animatePlaceholder,
            38
        );

        return;

    }


    characterIndex -= 1;

    input.placeholder =
        prompt.slice(
            0,
            characterIndex
        );


    if (
        characterIndex <= 0
    ) {

        deleting = false;

        promptIndex =
            (
                promptIndex + 1
            )
            % prompts.length;

        setTimeout(
            animatePlaceholder,
            450
        );

        return;

    }


    setTimeout(
        animatePlaceholder,
        18
    );

}


setTimeout(
    animatePlaceholder,
    500
);

</script>
        """,
        height=0,
        width=0,
    )


# =============================================================================
# HANDLE QUESTION
# =============================================================================

if (
    question
    and question.strip()
):

    clean_question = (
        question.strip()
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_question,
        }
    )


    with st.spinner(
        "Reviewing the evidence…"
    ):

        try:

            response = (
                answer_question(
                    clean_question,
                    source_count,
                )
            )

        except Exception as exc:

            st.error(
                "PRRISM couldn't "
                "complete that request."
            )

            # Development only.
            # Remove before public deployment.
            st.exception(
                exc
            )

            st.stop()


    st.session_state.messages.append(
        {
            "role": "assistant",
            "results": response[
                "results"
            ],
        }
    )

    st.rerun()


# =============================================================================
# ABOUT
# =============================================================================

if st.session_state.messages:

    with st.expander(
        "About these results"
    ):

        st.markdown(
            """
PRRISM retrieves evidence from a curated collection of
facial-recognition research, policy, and governance sources.

Results are presented in PRRISM's relevance order. Each explanation
is generated only from the retrieved evidence associated with that
source. The number of sources shown is selected by the researcher.
            """
        )