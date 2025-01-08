import os
import requests
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from litellm import completion
import openai
from openai import AsyncOpenAI

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
load_dotenv()


tracer.start()


@tracer.trace_llm(name="llm_call")
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini", name="default"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    
    actual_response = response.choices[0].message.content.strip()
    
    return actual_response


# Tools outside agents
@tracer.trace_tool(name="weather_tool")
def weather_tool(destination):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {"q": destination, "appid": api_key, "units": "metric"}
    print("Calculating weather for:", destination)
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        
        actual_result = f"{weather_description.capitalize()}, {temperature:.1f}°C"
        
        return actual_result
    except requests.RequestException:
        return "Weather data not available."


@tracer.trace_tool(name="currency_converter_tool")
def currency_converter_tool(amount, from_currency, to_currency):
    api_key = os.environ.get("EXCHANGERATE_API_KEY")
    base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()

        if data["result"] == "success":
            rate = data["conversion_rate"]
            actual_result = amount * rate
            
            return actual_result
        else:
            return None
    except requests.RequestException:
        return None


@tracer.trace_tool(name="flight_price_estimator_tool")
def flight_price_estimator_tool(origin, destination):
    # This is a mock function. In a real scenario, you'd integrate with a flight API.
    actual_result = f"Estimated price from {origin} to {destination}: $500-$1000"
    
    return actual_result


# Agent with persona
class ItineraryAgent:
    def __init__(self, persona="Itinerary Agent"):
        self.persona = persona

    @tracer.trace_agent(name="plan_itinerary")
    def plan_itinerary(self, user_preferences, duration=3):
        # Get weather information
        weather = weather_tool(user_preferences["destination"])

        # Get currency conversion if needed
        if "budget_currency" in user_preferences and user_preferences["budget_currency"] != "USD":
            budget = currency_converter_tool(
                user_preferences["budget"],
                user_preferences["budget_currency"],
                "USD"
            )
        else:
            budget = user_preferences["budget"]

        # Get flight price estimation
        flight_price = flight_price_estimator_tool(
            user_preferences["origin"],
            user_preferences["destination"]
        )

        # Prepare prompt for the LLM
        prompt = f"""As a {self.persona}, create a {duration}-day itinerary for a trip to {user_preferences['destination']}.
        Weather: {weather}
        Budget: ${budget}
        Flight Price: {flight_price}
        Preferences: {user_preferences.get('preferences', 'No specific preferences')}
        
        Please provide a detailed day-by-day itinerary."""

        # Get itinerary from LLM
        itinerary = llm_call(prompt)
        return itinerary


# Main function
@tracer.trace_agent(name="travel_agent")
def travel_agent():
    print("Welcome to the Personalized Travel Planner!\n")

    # Get user input
    # user_input = input("Please describe your ideal vacation: ")
    user_input = input("Please describe your ideal vacation: ")
    #"karela, 10 days, $100, nature"

    # Extract preferences
    preferences_prompt = f"""
Extract key travel preferences from the following user input:
"{user_input}"

Please provide the extracted information in this format:
Destination:
Activities:
Budget:
Duration (in days):
"""
    extracted_preferences = llm_call(preferences_prompt, name="extract_preferences")
    print("\nExtracted Preferences:")
    print(extracted_preferences)

    # Parse extracted preferences
    preferences = {}
    for line in extracted_preferences.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            preferences[key.strip()] = value.strip()

    # Validate extracted preferences
    required_keys = ["Destination", "Activities", "Budget", "Duration (in days)"]
    if not all(key in preferences for key in required_keys):
        print("\nCould not extract all required preferences. Please try again.")
        return

    # Fetch additional information
    weather = weather_tool(preferences["Destination"])
    print(f"\nWeather in {preferences['Destination']}: {weather}")

    # origin = input("Please enter your departure city: ")
    print("Please enter your departure city: ")
    origin = input("Please enter your departure city: ")
    # "delhi"
    flight_price = flight_price_estimator_tool(origin, preferences["Destination"])
    print(flight_price)

    # Plan itinerary
    itinerary_agent = ItineraryAgent()
    itinerary = itinerary_agent.plan_itinerary(
        {"destination": preferences["Destination"], "origin": origin, "budget": float(preferences["Budget"].replace("$", "")), "budget_currency": "USD"},
        int(preferences["Duration (in days)"])
    )
    print("\nPlanned Itinerary:")
    print(itinerary)

    # Currency conversion
    budget_amount = float(preferences["Budget"].replace("$", "").replace(",", ""))
    converted_budget = currency_converter_tool(budget_amount, "USD", "INR")
    if converted_budget:
        print(f"\nBudget in INR: {converted_budget:.2f} INR")
    else:
        print("\nCurrency conversion not available.")

    # Generate travel summary
    summary_prompt = f"""
Summarize the following travel plan:

Destination: {preferences['Destination']}
Activities: {preferences['Activities']}
Budget: {preferences['Budget']}
Duration: {preferences['Duration (in days)']} days
Itinerary: {itinerary}
Weather: {weather}
Flight Price: {flight_price}

Travel Summary:
"""
    travel_summary = llm_call(summary_prompt, name="generate_summary")
    print("\nTravel Summary:")
    print(travel_summary)


if __name__ == "__main__":
    travel_agent()
    tracer.stop()
