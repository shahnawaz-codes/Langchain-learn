from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()


# model = init_chat_model("huggingface:deepseek-ai/DeepSeek-R1")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of France?")
print(result.content)