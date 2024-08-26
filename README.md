# RegressedBeta

#### Video Demo:  https://youtu.be/07B-NjYMjtA

#### Description:
Description:
RegressedBeta is a Python program that calculates and visualizes the beta coefficient of a stock relative to a market benchmark using data from Yahoo Finance. This tool allows users to input a market benchmark (like SPY for the S&P 500) and individual stock tickers for analysis. Users can specify custom date ranges or predefined periods for the calculation. The program fetches historical price data, computes daily returns, performs linear regression, and provides a comprehensive analysis of the stock's beta.

#### Key Features:
Flexible time period selection (custom dates or predefined periods like '6mo', '1y', etc.)
Calculation of daily returns and beta coefficient
Statistical analysis including confidence intervals and p-values
Visualization of the regression line with confidence intervals
Detailed output of beta, alpha (intercept), and related statistics

#### Files:
The program is self-contained in a single Python script and can run independently. It requires several Python libraries including yfinance, pandas, numpy, matplotlib, seaborn, statsmodels, and colorama.

#### Process:
The development of RegressedBeta was inspired by my studies in the CFA program and a desire to apply statistical concepts to financial market analysis. The beta coefficient, a key measure of a stock's volatility relative to the market, is a fundamental concept in finance that I wanted to explore programmatically.
