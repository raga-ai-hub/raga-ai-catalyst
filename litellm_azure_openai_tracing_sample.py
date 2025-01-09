from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
from dotenv import load_dotenv
import requests

load_dotenv()

from litellm import completion

api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = "2024-05-01-preview"
api_key = os.getenv("AZURE_OPENAI_API_KEY")

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


@tracer.trace_llm(name="litellm_azure_llm_call")
def litellm_azure_llm_call(prompt, model="azure/gpt-4o-mini"):
    try:
        response = completion(
                model = model,             # model = azure/<your deployment name> 
                api_base = api_base,                                      # azure api base
                api_version = api_version,                                   # azure api version
                api_key = api_key,                                       # azure api key
                messages = [{"role": "user", "content": prompt}],
            )
        return response.choices[0].message.content.strip() 
   
    except Exception as e:
        print(f"Error in azure_llm_call: {str(e)}")
        raise


def main():
    try:
        response = litellm_azure_llm_call(
            prompt="Tell me a joke"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
    tracer.stop()
