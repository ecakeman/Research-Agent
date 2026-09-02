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
            "Judge support for the specific sub-question(s) this chunk covers, "
            "not whether the chunk answers the entire research question. JSON only: "
            '{{"chunk_id":"...","relevant":true,"support_level":"direct|partial|weak|none",'
            '"reason":"...","covers":["sub_question text"]}}. '
            "A chunk can be direct evidence for one sub-question even if it does not "
            "answer the whole research question. "
            "Only mark support_level=direct when the chunk explicitly supports the covered "
            "sub-question. Do not require the chunk to answer other sub-questions. "
            "covers must contain only sub_questions explicitly supported by the chunk.",
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
            "You are a query rewriting component for retrieval. "
            "The previous retrieval round was insufficient. Do not answer the question. "
            "1. Identify the unresolved sub-question(s). "
            "2. Identify the missing evidence needed to answer them. "
            "3. Generate a retrieval query specifically targeting that evidence. "
            'JSON only: {{"rewritten_query":"...","focus":["..."]}}. '
            "The rewritten query preserves original intent, keeps important technical terms, "
            "uses terms from the original question, sub-questions and observed evidence, "
            "targets the missing evidence, and avoids generic paraphrases. "
            "Do not invent facts, APIs, systems or answers. "
            "focus lists the concrete missing evidence or unresolved sub-question(s).",
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
