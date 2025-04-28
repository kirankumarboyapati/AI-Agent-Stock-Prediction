#!/usr/bin/env python3
"""
This Streamlit application fetches historical stock data and computes various risk metrics.
It performs portfolio risk breakdown analysis and scenario analysis.
It also passes the computed risk metrics and portfolio breakdown data to a Crew AI agent
to obtain an investment decision recommendation, which is then displayed in the Streamlit UI.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from yahooquery import Ticker
from dotenv import load_dotenv
from datetime import datetime
from textwrap import dedent

# CrewAI imports: Import the necessary classes for our Crew AI–based investment decision system.
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Update system path to import custom modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import custom tools for Crew AI.
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools

# Import the raw Yahoo Finance News tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool as RawYahooFinanceNewsTool

# Load environment variables (e.g., API keys) from a .env file.
load_dotenv()

# -----------------------------------------------------------------------------
# Wrapper for YahooFinanceNewsTool to satisfy Agent tool interface
# -----------------------------------------------------------------------------
class YahooFinanceNewsToolWrapper:
    def __init__(self):
        self.raw = RawYahooFinanceNewsTool()
        self.name = self.raw.name
        self.description = self.raw.description

    def func(self, query: str):
        # RawYahooFinanceNewsTool expects a dict with "query"
        return self.raw.run({"query": query})

# ------------------------------
# Data Fetching Functions
# ------------------------------
def fetch_stock_data(ticker_symbol, period='1y'):
    st.info(f"Fetching historical data for {ticker_symbol} (Period: {period})...")
    ticker = Ticker(ticker_symbol)
    data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
        data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert(None)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None
    
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

def fetch_current_stock_price(ticker_symbol):
    st.info(f"Fetching current stock price for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    try:
        realtime_data = ticker.price
        key = ticker_symbol.upper()
        if isinstance(realtime_data, dict):
            if key in realtime_data and isinstance(realtime_data[key], dict) and "regularMarketPrice" in realtime_data[key]:
                return realtime_data[key]["regularMarketPrice"]
            elif "regularMarketPrice" in realtime_data:
                return realtime_data["regularMarketPrice"]
        st.error("Failed to fetch current stock price from yahooquery.")
        return None
    except Exception as e:
        st.error(f"Error fetching current stock price: {e}")
        return None

# ------------------------------
# Risk Metrics Calculation Functions
# ------------------------------
def calculate_risk_metrics(data, confidence=0.05):
    data = data.sort_values(by='date').copy()
    data['returns'] = data['close'].pct_change()
    returns = data['returns'].dropna()
    
    var_value = np.percentile(returns, confidence * 100)
    
    data['cumulative_return'] = (1 + returns).cumprod()
    data['running_max'] = data['cumulative_return'].cummax()
    data['drawdown'] = (data['cumulative_return'] - data['running_max']) / data['running_max']
    max_drawdown = data['drawdown'].min()
    
    volatility = returns.std() * np.sqrt(252)
    
    risk_metrics = {
        "var": var_value,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }
    return risk_metrics, data

def calculate_scenario_risk_metrics(data, shock, confidence=0.05):
    data = data.sort_values(by='date').copy()
    data['returns'] = data['close'].pct_change()
    data['shock_returns'] = data['returns'] - shock
    shock_returns = data['shock_returns'].dropna()
    
    var_value = np.percentile(shock_returns, confidence * 100)
    data['cumulative_return'] = (1 + shock_returns).cumprod()
    data['running_max'] = data['cumulative_return'].cummax()
    data['drawdown'] = (data['cumulative_return'] - data['running_max']) / data['running_max']
    max_drawdown = data['drawdown'].min()
    volatility = shock_returns.std() * np.sqrt(252)
    
    scenario_metrics = {
        "var": var_value,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }
    return scenario_metrics

# ------------------------------
# Portfolio Breakdown Analysis
# ------------------------------
def analyze_portfolio_breakdown(portfolio_str, period='1y', confidence=0.05):
    lines = portfolio_str.strip().splitlines()
    records = []
    for line in lines:
        try:
            ticker, asset_class, position_size = [x.strip() for x in line.split(",")]
            position_size = float(position_size)
            data = fetch_stock_data(ticker, period)
            if data is None:
                continue
            metrics, _ = calculate_risk_metrics(data, confidence)
            records.append({
                "Ticker": ticker,
                "Asset Class": asset_class,
                "Position Size": position_size,
                "VaR": metrics["var"],
                "Max Drawdown": metrics["max_drawdown"],
                "Volatility": metrics["volatility"]
            })
        except Exception as e:
            st.error(f"Error processing line '{line}': {e}")
    
    if records:
        df = pd.DataFrame(records)
        grouped = df.groupby("Asset Class").apply(
            lambda g: pd.Series({
                "Total Position": g["Position Size"].sum(),
                "Weighted VaR": np.average(g["VaR"], weights=g["Position Size"]),
                "Weighted Max Drawdown": np.average(g["Max Drawdown"], weights=g["Position Size"]),
                "Weighted Volatility": np.average(g["Volatility"], weights=g["Position Size"])
            })
        ).reset_index()
        return df, grouped
    else:
        return None, None

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_price_chart(data, ticker_symbol):
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['close'], label='Close Price')
    ax.set_title(f"{ticker_symbol} Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_drawdown_chart(data, ticker_symbol):
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['drawdown'], label='Drawdown')
    ax.set_title(f"{ticker_symbol} Drawdown Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig)

def plot_return_histogram(returns, var_value):
    fig, ax = plt.subplots()
    ax.hist(returns, bins=50, alpha=0.7, label='Daily Returns')
    ax.axvline(var_value, linestyle='dashed', linewidth=2, label=f'VaR ({var_value:.2%})')
    ax.set_title("Histogram of Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# ------------------------------
# Crew AI Integration for Investment Decision
# ------------------------------
gpt_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o"
)

class RiskMetricsAnalysisAgents:
    def risk_metrics_investment_advisor(self):
        return Agent(
            llm=gpt_model,
            role="Risk Metrics Investment Advisor",
            goal="Provide actionable investment recommendations based on risk metrics and portfolio breakdown analysis.",
            backstory="You are an experienced risk analyst specializing in portfolio risk metrics and diversification. Analyze the provided risk metrics from single ticker analysis and aggregated portfolio breakdown data to give a clear investment recommendation (BUY, SELL, or HOLD) with detailed reasoning.",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k,
                YahooFinanceNewsToolWrapper()    # ← wrapped instance
            ]
        )

    def risk_metrics_analysis(self, agent, risk_metrics, portfolio_breakdown):
        breakdown_text = ""
        if portfolio_breakdown is not None and not portfolio_breakdown.empty:
            breakdown_text = portfolio_breakdown.to_csv(index=False)
        
        description = dedent(f"""
            Analyze the following risk metrics from a single ticker analysis:
            - Value at Risk (VaR): {risk_metrics['var']:.2%}
            - Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}
            - Annualized Volatility: {risk_metrics['volatility']:.2%}
            
            Additionally, consider the following portfolio breakdown by asset class (weighted averages):
            {breakdown_text}
            
            Based on these values and current market conditions, provide a very detailed investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning and explanation.
        """)
        
        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis with a clear investment recommendation (BUY, SELL, or HOLD) and supporting reasoning based on the provided risk metrics and portfolio breakdown."
        )

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Portfolio Risk Assessment Dashboard")
    st.write(
        "Visualize portfolio risk metrics, get detailed breakdowns by asset class and position, "
        "simulate market scenarios, and obtain an investment decision recommendation via Crew AI."
    )
    
    # Sidebar: Single Ticker Analysis Inputs
    st.sidebar.header("Single Ticker Analysis")
    ticker_symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.sidebar.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
    confidence_level = st.sidebar.slider(
        "VaR Confidence Level",
        min_value=0.01, max_value=0.1, value=0.05, step=0.01,
        help="Lower tail probability for VaR calculation (e.g., 0.05 for 5%)"
    )
    
    # Section 1: Single Ticker Analysis
    if st.button("Fetch and Analyze Data"):
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Historical Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            plot_price_chart(data, ticker_symbol)
            
            risk_metrics, risk_data = calculate_risk_metrics(data, confidence=confidence_level)
            st.subheader("Calculated Risk Metrics")
            st.markdown(f"- **Value at Risk (VaR) at {confidence_level*100:.0f}% level:** {risk_metrics['var']:.2%}")
            st.markdown(f"- **Maximum Drawdown:** {risk_metrics['max_drawdown']:.2%}")
            st.markdown(f"- **Annualized Volatility:** {risk_metrics['volatility']:.2%}")
            
            plot_drawdown_chart(risk_data, ticker_symbol)
            returns = risk_data['returns'].dropna()
            plot_return_histogram(returns, risk_metrics['var'])
            
            st.session_state.risk_metrics = risk_metrics
            st.session_state.single_data = data
    
    # Section 2: Portfolio Risk Breakdown Analysis
    st.header("Portfolio Risk Breakdown")
    st.write(
        "Enter your portfolio positions in the format: **Ticker, Asset Class, Position Size** (one per line). For example:\n\n"
        "`AAPL, Equity, 100`\n`MSFT, Equity, 150`\n`TLT, Bond, 200`\n`GLD, Commodity, 50`"
    )
    portfolio_input = st.text_area(
        "Portfolio Positions", 
        value="AAPL, Equity, 100\nMSFT, Equity, 150\nTLT, Bond, 200\nGLD, Commodity, 50", 
        height=150
    )
    if st.button("Analyze Portfolio Breakdown"):
        details_df, breakdown_df = analyze_portfolio_breakdown(
            portfolio_input,
            period=period_option,
            confidence=confidence_level
        )
        if details_df is not None:
            st.subheader("Individual Position Risk Metrics")
            st.dataframe(details_df)
            st.subheader("Risk Breakdown by Asset Class (Weighted Averages)")
            st.dataframe(breakdown_df)
            st.session_state.breakdown_df = breakdown_df
        else:
            st.error("No valid portfolio positions found or error in processing.")
    
    # Section 3: Scenario Analysis
    st.header("Scenario Analysis: Validate Risk Metrics")
    shock_percent = st.slider(
        "Simulated Market Shock (%)",
        min_value=0.0, max_value=10.0, value=2.0, step=0.5,
        help="Enter the percentage shock to subtract from daily returns (e.g., 2% shock)."
    )
    if st.button("Validate Risk Metrics Under Scenario"):
        if "single_data" not in st.session_state:
            st.error("Please perform the single ticker analysis first.")
        else:
            shock = shock_percent / 100.0
            base_metrics, base_data = calculate_risk_metrics(
                st.session_state.single_data,
                confidence=confidence_level
            )
            scenario_metrics = calculate_scenario_risk_metrics(
                st.session_state.single_data,
                shock,
                confidence=confidence_level
            )
            st.subheader("Baseline vs. Scenario Risk Metrics")
            comparison = pd.DataFrame({
                "Metric": ["VaR", "Max Drawdown", "Annualized Volatility"],
                "Baseline": [
                    f"{base_metrics['var']:.2%}",
                    f"{base_metrics['max_drawdown']:.2%}",
                    f"{base_metrics['volatility']:.2%}"
                ],
                f"Scenario (Shock -{shock_percent:.0f}%)": [
                    f"{scenario_metrics['var']:.2%}",
                    f"{scenario_metrics['max_drawdown']:.2%}",
                    f"{scenario_metrics['volatility']:.2%}"
                ]
            })
            st.dataframe(comparison)
    
    # Section 4: Investment Decision via Crew AI Agent
    st.header("Investment Decision Recommendation via Crew AI")
    if st.button("Get Investment Decision"):
        if "risk_metrics" in st.session_state and "breakdown_df" in st.session_state:
            risk_metrics = st.session_state.risk_metrics
            portfolio_breakdown = st.session_state.breakdown_df
            
            agents = RiskMetricsAnalysisAgents()
            advisor_agent = agents.risk_metrics_investment_advisor()
            analysis_task = agents.risk_metrics_analysis(advisor_agent, risk_metrics, portfolio_breakdown)
            
            crew = Crew(
                agents=[advisor_agent],
                tasks=[analysis_task],
                verbose=True
            )
            result = crew.kickoff()
            st.subheader("Crew AI Investment Decision")
            st.write(result)
        else:
            st.error("Please complete both the Single Ticker Analysis and Portfolio Risk Breakdown Analysis first.")

if __name__ == '__main__':
    main()
