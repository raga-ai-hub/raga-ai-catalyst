import os
os.chdir('..')

import random
from textblob import TextBlob
import openai
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers import Tracer

# Load environment variables
load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
catalyst = RagaAICatalyst(
    access_key="access_key",
    secret_key="secret_key",
    base_url="base_url"
)

# Start tracing
tracer = Tracer(
    project_name="prompt_metric_dataset",
    dataset_name="testing-1",
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
class FinancialAnalysisSystem:
    def __init__(self):
        self.stock_data = {}
        self.news_sentiment = {}
        self.economic_indicators = {}

    @tracer.trace_tool(name="fetch_stock_data")
    def fetch_stock_data(self, symbol):
        print("Fetching stock data...")
        return {
            "symbol": symbol,
            "price": round(random.uniform(50, 500), 2),
            "change": round(random.uniform(-5, 5), 2),
        }

    @tracer.trace_tool(name="fetch_news_articles")
    def fetch_news_articles(self, company):
        return [
            f"{company} announces new product line",
            f"{company} reports quarterly earnings",
            f"{company} faces regulatory scrutiny",
        ]

    @tracer.trace_tool(name="analyze_sentiment")
    def analyze_sentiment(self, text):
        return TextBlob(text).sentiment.polarity

    @tracer.trace_tool(name="fetch_economic_indicators")
    def fetch_economic_indicators(self):
        input("Press Enter to fetch economic indicators...")
        print("Fetching economic indicators...")
        return {
            "gdp_growth": round(random.uniform(-2, 5), 2),
            "unemployment_rate": round(random.uniform(3, 10), 2),
            "inflation_rate": round(random.uniform(0, 5), 2),
        }

    @tracer.trace_llm(name="analyze_market_conditions")
    def analyze_market_conditions(self, stock_data, sentiment, economic_indicators):
        prompt = f"""
        Analyze the following market conditions and provide a brief market outlook:
        Stock: {stock_data['symbol']} at ${stock_data['price']} (change: {stock_data['change']}%)
        News Sentiment: {sentiment}
        Economic Indicators:
        - GDP Growth: {economic_indicators['gdp_growth']}%
        - Unemployment Rate: {economic_indicators['unemployment_rate']}%
        - Inflation Rate: {economic_indicators['inflation_rate']}%
        """
        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    @tracer.trace_llm(name="generate_investment_recommendation")
    def generate_investment_recommendation(self, market_outlook, risk_tolerance):
        prompt = f"""
        Based on the following market outlook and investor risk tolerance,
        provide a specific investment recommendation:
        Market Outlook: {market_outlook}
        Investor Risk Tolerance: {risk_tolerance}
        """
        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    @tracer.trace_agent(name="financial_advisor_agent")
    def financial_advisor_agent(self, stock_symbol, risk_tolerance):
        self.stock_data = self.fetch_stock_data(stock_symbol)
        news_articles = self.fetch_news_articles(stock_symbol)
        sentiment_scores = [self.analyze_sentiment(article) for article in news_articles]
        self.news_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        self.economic_indicators = self.fetch_economic_indicators()
        market_outlook = self.analyze_market_conditions(
            self.stock_data, self.news_sentiment, self.economic_indicators
        )
        recommendation = self.generate_investment_recommendation(market_outlook, risk_tolerance)
        return recommendation

    @tracer.trace_tool(name="run_analysis")
    def run_analysis(self, stock_symbol, risk_tolerance):
        recommendation = self.financial_advisor_agent(stock_symbol, risk_tolerance)
        print(f"\nAnalysis for {stock_symbol}:")
        print(f"Stock Data: {self.stock_data}")
        print(f"News Sentiment: {self.news_sentiment}")
        print(f"Economic Indicators: {self.economic_indicators}")
        print(f"\nInvestment Recommendation:\n{recommendation}")
        if "buy" in recommendation.lower():
            self.execute_buy_order(stock_symbol)
        elif "sell" in recommendation.lower():
            self.execute_sell_order(stock_symbol)
        else:
            print("No action taken based on the current recommendation.")

    @tracer.trace_tool(name="execute_buy_order")
    def execute_buy_order(self, symbol):
        print(f"Executing buy order for {symbol}")

    @tracer.trace_tool(name="execute_sell_order")
    def execute_sell_order(self, symbol):
        print(f"Executing sell order for {symbol}")
   
if __name__ == "__main__":     
    # Create an instance of FinancialAnalysisSystem
    analysis_system = FinancialAnalysisSystem()

    # Run an analysis for Apple stock with moderate risk tolerance
    analysis_system.run_analysis("AAPL", "moderate")

    # Stop the tracer when analysis is complete
    tracer.stop()