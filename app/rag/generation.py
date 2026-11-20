
import time

from langchain_core.prompts import ChatPromptTemplate

from app.core.logger import get_logger
from app.core.exceptions import GenerationError
from app.infrastructure.llm_gateway import get_llm

logger = get_logger(__name__)

PROMPT = ChatPromptTemplate.from_template("""
You are a document-based question answering assistant.

Answer the question using only the provided context.

Rules:
- Do not use external knowledge.
- Do not invent facts or information.
- If the answer is not available in the context, say:
  "The uploaded document does not contain sufficient information to answer this question."
- Give a clear and concise answer.
- Include relevant page numbers when available.

Context:
{context}

Question:
{question}

Answer:
""")


def generate_answer(question: str, documents: list):
    generation_start = None

    try:
        if not documents:
            raise GenerationError("No context documents provided")

        context = "\n\n".join(
            f"Page {doc.metadata.get('page_number', 'Unknown')}:\n{doc.page_content}"
            for doc in documents
        )

        chain = PROMPT | get_llm()

        generation_start = time.perf_counter()

        response = chain.invoke({
            "context": context,
            "question": question
        })

        latency_ms = (time.perf_counter() - generation_start) * 1000

        usage = response.usage_metadata or response.response_metadata.get(
            "token_usage", {}
        )

        input_tokens = usage.get(
            "input_tokens", usage.get("prompt_tokens", 0)
        )
        output_tokens = usage.get(
            "output_tokens", usage.get("completion_tokens", 0)
        )

        logger.info(
            "LLM generation completed | model=%s | latency=%.2f ms | input_tokens=%s | output_tokens=%s",
            response.response_metadata.get("model_name", "configured-model"),
            latency_ms,
            input_tokens,
            output_tokens
        )

        return response.content

    except GenerationError:
        raise

    except Exception as e:
        if generation_start is not None:
            logger.exception(
                "LLM generation failed | latency=%.2f ms",
                (time.perf_counter() - generation_start) * 1000
            )
        else:
            logger.exception("Answer generation failed")

        raise GenerationError("Failed to generate answer") from e
