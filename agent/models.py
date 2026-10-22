from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model("openai/gpt-oss-20b",
                         model_provider="groq")

response = model.invoke("Tell me something about the sun in 2 sentences.")
print(response.content)