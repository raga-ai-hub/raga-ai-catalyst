import litellm
from litellm import completion
import os

api_base="https://dsragaai.openai.azure.com/openai/deployments/azure-gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
model="azure/gpt-4o-mini"
api_version="2024-05-01-preview"
api_key="49024331d17d4e79bbc729718afe1da2"
response = litellm.completion(
    model = model,             # model = azure/<your deployment name> 
    api_base = api_base,                                      # azure api base
    api_version = api_version,                                   # azure api version
    api_key = api_key,                                       # azure api key
    messages = [{"role": "user", "content": "good morning"}],
)
print(response)
