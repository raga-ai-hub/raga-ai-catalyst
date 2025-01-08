from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic
import google.generativeai as genai
from litellm import completion as litellm_completion
from litellm import acompletion as litellm_acompletion
from groq import Groq, AsyncGroq
# from llama_cpp import Llama

# Initialize providers
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

# OpenAI implementations
@tracer.trace_llm(name="openai_llm_call")
def openai_llm_call(prompt, max_tokens=512, model="gpt-3.5-turbo", temperature=0.7, gt=None):
    print(f"OpenAI sync call started with model: {model}")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print("OpenAI sync call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in openai_llm_call: {str(e)}")
        raise

@tracer.trace_llm(name="openai_llm_call_async")
async def openai_llm_call_async(prompt, max_tokens=512, model="gpt-3.5-turbo", temperature=0.7, gt=None):
    print(f"OpenAI async call started with model: {model}")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print("OpenAI async call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in openai_llm_call_async: {str(e)}")
        raise

# Anthropic implementations
@tracer.trace_llm(name="anthropic_llm_call")
def anthropic_llm_call(prompt, max_tokens=512, model="claude-3-sonnet-20240229", temperature=0.7, gt=None):
    print(f"Anthropic sync call started with model: {model}")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        print("Anthropic sync call completed successfully")
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error in anthropic_llm_call: {str(e)}")
        raise

@tracer.trace_llm(name="anthropic_llm_call_async")
async def anthropic_llm_call_async(prompt, max_tokens=512, model="claude-3-sonnet-20240229", temperature=0.7, gt=None):
    print(f"Anthropic async call started with model: {model}")
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        print("Anthropic async call completed successfully")
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error in anthropic_llm_call_async: {str(e)}")
        raise

# Google/Gemini implementations
@tracer.trace_llm(name="gemini_llm_call")
def gemini_llm_call(prompt, max_tokens=512, temperature=0.7, top_p=1, top_k=40, gt=None):
    print("Gemini sync call started")
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ))
        print("Gemini sync call completed successfully")
        return response.text.strip()
    except Exception as e:
        print(f"Error in gemini_llm_call: {str(e)}")
        raise

@tracer.trace_llm(name="gemini_llm_call_async")
async def gemini_llm_call_async(prompt, max_tokens=512, temperature=0.7, top_p=1, top_k=40, gt=None):
    print("Gemini async call started")
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = await model.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ))
        print("Gemini async call completed successfully")
        return response.text.strip()
    except Exception as e:
        print(f"Error in gemini_llm_call_async: {str(e)}")
        raise

# LiteLLM implementations
@tracer.trace_llm(name="litellm_call")
def litellm_llm_call(prompt, max_tokens=512, model="gpt-3.5-turbo", temperature=0.7, gt=None):
    print(f"LiteLLM sync call started with model: {model}")
    try:
        response = litellm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        print("LiteLLM sync call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in litellm_llm_call: {str(e)}")
        raise

@tracer.trace_llm(name="litellm_call_async")
async def litellm_llm_call_async(prompt, max_tokens=512, model="gpt-3.5-turbo", temperature=0.7, gt=None):
    print(f"LiteLLM async call started with model: {model}")
    try:
        response = await litellm_acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        print("LiteLLM async call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in litellm_llm_call_async: {str(e)}")
        raise

# Groq implementations
@tracer.trace_llm(name="groq_llm_call")
def groq_llm_call(prompt, max_tokens=512, model="mixtral-8x7b-32768", temperature=0.7, gt=None):
    print(f"Groq sync call started with model: {model}")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print("Groq sync call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in groq_llm_call: {str(e)}")
        raise

@tracer.trace_llm(name="groq_llm_call_async")
async def groq_llm_call_async(prompt, max_tokens=512, model="mixtral-8x7b-32768", temperature=0.7, gt=None):
    print(f"Groq async call started with model: {model}")
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print("Groq async call completed successfully")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in groq_llm_call_async: {str(e)}")
        raise


async def async_main():
    prompt = "What is the meaning of life?"
    try:
        # OpenAI async
        openai_response = await openai_llm_call_async(prompt)
        print("Async OpenAI Response:", openai_response)
        
        # Anthropic async
        anthropic_response = await anthropic_llm_call_async(prompt)
        print("Async Anthropic Response:", anthropic_response)
        
        # Gemini async
        gemini_response = await gemini_llm_call_async(prompt)
        print("Async Gemini Response:", gemini_response)
        
        # LiteLLM async
        litellm_response = await litellm_llm_call_async(prompt)
        print("Async LiteLLM Response:", litellm_response)
        
        # Groq async
        groq_response = await groq_llm_call_async(prompt)
        print("Async Groq Response:", groq_response)
        
    except Exception as e:
        print(f"Error in async_main: {str(e)}")

def main():
    prompt = "How are you?"
    try:
        # Sync calls
        openai_response = openai_llm_call(prompt)
        print("OpenAI Response:", openai_response)
        
        anthropic_response = anthropic_llm_call(prompt)
        print("Anthropic Response:", anthropic_response)
        
        gemini_response = gemini_llm_call(prompt)
        print("Gemini Response:", gemini_response)
        
        litellm_response = litellm_llm_call(prompt)
        print("LiteLLM Response:", litellm_response)
        
        groq_response = groq_llm_call(prompt)
        print("Groq Response:", groq_response)
        
        # Run async calls
        asyncio.run(async_main())
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
    tracer.stop()