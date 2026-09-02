from langchain_core.prompts import ChatPromptTemplate

ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze a technical research question. Reply with JSON only: "
            '{{"intent":"fact|comparison|explanation|multi_hop|how_to|unknown",'
            '"entities":["..."],"sub_questions":["..."]}}. '
            "Keep technical terms. Do not invent facts outside the question.",
        ),
        ("human", "Question: {query}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Judge whether a chunk supports the research question. JSON only: "
            '{{"chunk_id":"...","relevant":true,"support_level":"direct|partial|weak|none",'
            '"reason":"...","covers":["sub_question text"]}}. '
            "covers must be a subset of the given sub_questions. "
            "direct means the chunk explicitly answers that sub-question.",
        ),
        (
            "human",
            "Question: {query}\nSub-questions: {sub_questions}\n"
            "chunk_id: {chunk_id}\nChunk:\n{chunk}",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriting component for retrieval. First identify the evidence gaps, "
            "then write a targeted search query that would retrieve documents filling those gaps. "
            'JSON only: {{"rewritten_query":"...","focus":["..."]}}. '
            "focus must list the concrete missing sub-questions or evidence gaps. "
            "You may extract retrieval terms from the original question, sub-questions, and observed evidence "
            "(titles, sections, quotes). Do not guess answers. Do not invent facts, APIs, or systems "
            "that do not appear in the inputs.",
        ),
        (
            "human",
            "Original question: {original_query}\n"
            "Sub-questions: {sub_questions}\n"
            "Evidence summary:\n{evidence_summary}\n"
            "Missing sub-questions: {missing_sub_questions}\n"
            "Evidence gaps: {evidence_gaps}\n"
            "Failure reasons: {failure_reasons}",
        ),
    ]
)

COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract evidence items from the chunk. JSON only: "
            '{{"claim":"...","quote":"verbatim substring of the chunk"}}. '
            "quote MUST be copied from the chunk text. Never invent a quote.",
        ),
        ("human", "Question: {query}\nChunk:\n{chunk}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the provided evidence. If evidence is insufficient, say so. "
            "Do not add facts that are not in evidence. Do not invent URLs. "
            "Each key external fact must bind to a citation chunk_id from evidence. "
            "JSON only: "
            '{{"answer":"...","citations":[{{"chunk_id":"...","claim_index":0}}]}}',
        ),
        ("human", "Question: {query}\nEvidence:\n{evidence}"),
    ]
)
