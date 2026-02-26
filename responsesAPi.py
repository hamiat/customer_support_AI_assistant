import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-03-01-preview",
)

def chat():
    response = client.responses.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input="If you are stranded on an island, what three items would you want to have with you for entertainment?")

    print(response.output_text)

if __name__ == "__main__":
    chat()