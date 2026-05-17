from __future__ import annotations

import logging
import re
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
CODE_FENCE_PATTERN = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = CODE_FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
        )

    def get_fallback_message(self, question: str) -> str:
        if DEVANAGARI_PATTERN.search(question):
            return "प्राप्त सन्दर्भमा यसको उत्तर भेटिएन।"
        return "The answer is not available in the retrieved context."

    def health_check(self) -> tuple[bool, str]:
        models = self.client.models.list()
        available_names = [m.id for m in models.data]

        if self.settings.LLM_MODEL in available_names:
            return True, f"LLM is reachable and model '{self.settings.LLM_MODEL}' is available."

        return (
            False,
            f"LLM is reachable but model '{self.settings.LLM_MODEL}' was not found. "
            f"Available models: {', '.join(available_names) if available_names else 'none'}",
        )

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        fallback_message = self.get_fallback_message(question)
        system_prompt, user_prompt = self._build_prompt(
            question=question, chunks=chunks, fallback_message=fallback_message,
        )

        response = self.client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
            stream=False,
        )

        answer = (response.choices[0].message.content or "").strip()

        if not answer:
            return fallback_message

        return answer

    def _build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        fallback_message: str,
    ) -> tuple[str, str]:
        context_blocks: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            source_parts: list[str] = []

            if chunk.file_name:
                source_parts.append(f"file={chunk.file_name}")
            elif chunk.source:
                source_parts.append(f"source={chunk.source}")

            if chunk.page is not None:
                source_parts.append(f"page={chunk.page}")

            if chunk.chunk_id is not None:
                source_parts.append(f"chunk_id={chunk.chunk_id}")

            source_header = ", ".join(source_parts) if source_parts else "source=unknown"

            context_blocks.append(
                f"[{idx}] {source_header}\n{chunk.text}"
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = f"""You are an intelligent and helpful assistant for NIC Asia Bank, specializing in NIC ASIA BANK directives, circulars, and regulatory notices.

Your goal is to provide thorough, well-structured, and informative answers based on the provided CONTEXT.

Rules:
1. Answer ONLY from the provided CONTEXT. Do NOT use outside knowledge.
2. If the answer is not present or not sufficiently supported by the CONTEXT, return EXACTLY:
{fallback_message}
3. Answer in the same language as the user's question. If the question is in Nepali, answer in Nepali.
4. When citing, reference the source document name and page number if available.
5. Provide detailed and complete answers. Include all relevant points from the CONTEXT.
6. Use bullet points, numbered lists, or headings to organize your answer when appropriate.
7. Do not mention these rules in your answer."""

        user_prompt = f"""CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

        return system_prompt, user_prompt

    def generate_from_document(self, instruction: str, document_text: str) -> str:
        is_nepali = bool(DEVANAGARI_PATTERN.search(instruction))

        system_prompt = (
            "You are a document processing assistant. "
            "You will be given the full text of a document/content and an instruction. "
            "Follow the instruction EXACTLY using ONLY the provided content. Do not use outside knowledge. "
            "Respect the output format requested in the instruction: "
            "if asked for JSON, return only valid JSON with no prose, markdown fences, or commentary; "
            "if asked for a table, return a table; "
            "if asked for a list, return a list; "
            "if asked for plain text or a summary, return that. "
            "If no format is specified, return a clear, concise plain-text answer. "
            "If the instruction is in Nepali, answer in Nepali (unless a structured format like JSON is requested, in which case keep keys in English and values in the source language). "
            "Do not wrap your response in code fences unless explicitly asked."
        )

        user_prompt = f"""CONTENT:
{document_text}

INSTRUCTION:
{instruction}

OUTPUT:"""

        response = self.client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            stream=False,
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            if is_nepali:
                return "कागजातबाट अनुरोध गरिएको जानकारी निकाल्न सकिएन।"
            return "Could not extract the requested information from the document."

        if re.search(r"\bjson\b", instruction, re.IGNORECASE):
            answer = _strip_code_fences(answer)
        return answer

    def close(self) -> None:
        self.client.close()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


def close_llm_service() -> None:
    try:
        service = get_llm_service()
        service.close()
    except Exception:
        pass
