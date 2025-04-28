#!/usr/bin/env python3
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import pandas as pd
import numpy as np
import sys
import json

# ensure we can import BaseAgent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.Agents.base_agent import BaseAgent

# CrewAI + LLM
import crewai
from crewai import Task, Crew, Process
from langchain_openai import ChatOpenAI

# ------------------------------
# Risk Metric Calculator (rolling)
# ------------------------------
def add_rolling_risk_metrics(df: pd.DataFrame, window: int, confidence: float):
    df = df.sort_values('date').copy()
    df['returns'] = df['close'].pct_change()
    # rolling VaR
    df['rolling_var'] = df['returns'].rolling(window).quantile(confidence)
    # rolling volatility
    df['rolling_volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    # rolling drawdown
    df['cum_return'] = (1 + df['returns']).cumprod()
    df['rolling_max'] = df['cum_return'].cummax()
    df['rolling_drawdown'] = (df['cum_return'] - df['rolling_max']) / df['rolling_max']
    return df.dropna()

# ------------------------------
# CrewAI agent for risk-based signals
# ------------------------------
class RiskBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"Risk-based trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on rolling risk metrics",
            backstory="You are an expert risk-management quant. Use rolling VaR, volatility, and drawdown to decide.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized RiskBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description="""
The global pandas DataFrame `data` has columns:
  date, high, low, close,
  returns, rolling_var, rolling_volatility, rolling_drawdown.

For each row, output exactly one of: BUY, SELL, or HOLD.
Return **only** a pure JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD,
with no additional commentary.
""",
            agent=self,
            expected_output="Pure JSON dict mapping YYYY-MM-DD to BUY/SELL/HOLD."
        )

# single shared LLM
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Rolling Risk Indicator for Backtrader
# ----------------------------------------
class RollingRiskBT(bt.Indicator):
    lines = ('rolling_var', 'rolling_volatility', 'rolling_drawdown',)
    params = (('window', 20), ('confidence', 0.05),)

    def __init__(self):
        self.addminperiod(self.p.window)

    def once(self, start, end):
        size = len(self.data)
        df = pd.DataFrame({
            'high':  [self.data.high[i] for i in range(size)],
            'low':   [self.data.low[i] for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
            'date':  pd.date_range(end=datetime.today(), periods=size, freq='D')
        })
        res = add_rolling_risk_metrics(df, self.p.window, self.p.confidence)
        for i in range(len(res)):
            self.lines.rolling_var[i]        = res['rolling_var'].iat[i]
            self.lines.rolling_volatility[i] = res['rolling_volatility'].iat[i]
            self.lines.rolling_drawdown[i]   = res['rolling_drawdown'].iat[i]

# ----------------------------------------
# Strategy driven by CrewAI signals
# ----------------------------------------
class RiskStrategy(bt.Strategy):
    params = (
        ('allocation', 1.0),
        ('signals', {}),
    )

    def __init__(self):
        self.signals = self.p.signals

    def next(self):
        dt = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        sig = self.signals.get(dt, 'HOLD')
        price = self.data.close[0]

        if sig == 'BUY' and not self.position:
            size = int((self.broker.getcash() * self.p.allocation) // price)
            if size:
                self.buy(size=size)
        elif sig == 'SELL' and self.position:
            self.sell(size=self.position.size)

# ----------------------------------------
# Backtest runner
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **kwargs)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    strat = cerebro.run()[0]
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    perf = {
        "Sharpe Ratio":   strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "Total Return %": r.get('rtot', 0) * 100,
        "Max Drawdown %": d.get('drawdown', 0) * 100
    }
    fig = cerebro.plot(iplot=False)[0][0]
    return perf, strat, fig

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    st.title("Risk Assessment Backtest With CrewAI Signals")

    st.sidebar.header("Parameters")
    ticker      = st.sidebar.text_input("Ticker", "AAPL")
    start       = st.sidebar.date_input("Start", datetime(2020, 1, 1))
    end         = st.sidebar.date_input("End",   datetime.today())
    window      = st.sidebar.number_input("Risk Window (days)", 20, step=1)
    confidence  = st.sidebar.slider("VaR Confidence", 0.01, 0.1, 0.05, 0.01)
    cash        = st.sidebar.number_input("Cash",       10000)
    comm        = st.sidebar.number_input("Commission", 0.001, step=0.0001)

    if st.sidebar.button("Run Backtest"):
        # 1) Fetch data via YahooQuery
        from yahooquery import Ticker as YQTicker
        df = YQTicker(ticker).history(period=None, start=start, end=end).reset_index()
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)

        # 2) Add rolling risk metrics
        risk_df = add_rolling_risk_metrics(df[['date', 'high', 'low', 'close']], window, confidence)

        # 3) Ask CrewAI for signals
        globals()['data'] = risk_df
        agent = RiskBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        crew.kickoff()
        raw     = task.output.raw    # <— use .raw instead of nonexistent .result
        signals = json.loads(raw)

        st.subheader("CrewAI Signals (raw JSON)")
        st.code(raw, language="json")

        # 4) Backtest
        feed = bt.feeds.PandasData(dataname=df.set_index('date'), fromdate=start, todate=end)
        perf, strat, fig = run_backtest(RiskStrategy, feed, cash=cash, commission=comm, signals=signals)

        # 5) Display
        st.subheader("Performance")
        st.write(perf)
        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
