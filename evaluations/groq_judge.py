
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        self.model = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            temperature=0,
            max_retries=1,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            result = self.model.with_structured_output(
                schema,
                method="function_calling",
            ).invoke(prompt)
            return result

        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            result = await self.model.with_structured_output(
                schema,
                method="function_calling",
            ).ainvoke(prompt)
            return result

        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq openai/gpt-oss-20b"