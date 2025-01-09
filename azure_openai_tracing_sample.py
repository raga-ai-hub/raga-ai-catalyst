from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
from dotenv import load_dotenv
import requests

load_dotenv()

from litellm import completion

# Initialize Azure OpenAI API client
import openai

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")  # Example: https://<your-resource-name>.openai.azure.com/
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Catalyst and Tracer
catalyst = RagaAICatalyst(
    access_key="access_key",
    secret_key="secret_key",
    base_url="base_url"
)
# Initialize tracer
tracer = Tracer(
    project_name="project_name",
    dataset_name="dataset_name",
    tracer_type="tracer_type",
    metadata={
        "model": "gpt-4",
        "environment": "production"
    },
    pipeline={
        "llm_model": "gpt-4",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)

tracer.start()


@tracer.trace_llm(name="azure_llm_call")
def azure_llm_call(prompt, max_tokens=512, model="gpt-4o-mini", deployment_name="azure-gpt-4o-mini"):
    """
    Azure OpenAI LLM call wrapped with tracing.
    """
    print(f"Prompt: {prompt}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Model: {model}")
    print(f"Deployment: {deployment_name}")

    try:
        # Azure OpenAI call
        response = openai.chat.completions.create(
            model=deployment_name,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,  
            stop=None,  
            stream=False,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in azure_llm_call: {str(e)}")
        raise


def main():
    try:
        response = azure_llm_call(
            prompt="How does Azure OpenAI Service integrate with other Azure products?"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
    tracer.stop()
