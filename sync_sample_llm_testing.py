from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from litellm import completion
import openai
from openai import OpenAI

catalyst = RagaAICatalyst(
    access_key="access_key",
    secret_key="secret_key",
    base_url="base_url"
)
# Initialize tracer
tracer = Tracer(
    project_name="prompt_metric_dataset",
    dataset_name="ChatOpenAI",
    tracer_type="anything",
    metadata={
        "model": "gpt-3.5-turbo",
        "environment": "production"
    },
    pipeline={
        "llm_model": "gpt-3.5-turbo",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)

tracer.start()


@tracer.trace_llm(name="llm_call")
def llm_call(prompt, max_tokens=512, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"Prompt: {prompt}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Model: {model}")
    input("Press Enter to continue...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in llm_call: {str(e)}")
        raise


def main():
    try:
        response = llm_call("how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
    tracer.stop()