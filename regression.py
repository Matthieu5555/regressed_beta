import re

import yfinance as yf

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy.stats import t

from colorama import Fore, Style

def main():
    """
    Main function to orchestrate the flow of the program.
    """
    market_ticker = input("Enter the benchmark ticker (SPY for example): ")
    
    while True:
        stock_ticker = input("Enter the stock ticker (or type 'NULL' to stop): ")
        if stock_ticker.upper() == 'NULL':
            break
        
        period, start_date, end_date = get_specific_period(stock_ticker)
        
        stock_data = get_historical_data(stock_ticker, start_date, end_date, period)
        market_data = get_historical_data(market_ticker, start_date, end_date, period)
        
        if stock_data.empty or market_data.empty:
            print("Insufficient data for the provided tickers.")
            continue
        
        stock_returns = calculate_daily_returns(stock_data)
        market_returns = calculate_daily_returns(market_data)
        
        returns = align_returns(stock_returns, market_returns)
        
        if returns.empty or len(returns) < 30:
            print("Not enough overlapping data points to compute beta.")
            continue
        
        model = perform_regression(returns)
        
        plot_regression(returns, model, stock_ticker, market_ticker)
        
        result = compute_trailing_beta(model)
        
        print(f'Trailing beta of {stock_ticker} relative to {market_ticker} during the selected period: {result["beta"]}')
        print(f'Intercept: {result["intercept"]}')
        print('Confidence Intervals:')
        print(result['confidence_intervals'])
        print('P-values:')
        print(result['p_values'])

def get_specific_period(ticker):
    """
    Prompts the user to input a specific date range or period.
    """
    date_regex = r"^\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}$"
    period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    while True:
        date_range = input("Enter the period (e.g. yyyy-mm-dd to yyyy-mm-dd or 6mo, 8y, etc.): ")
        if re.match(date_regex, date_range):
            start_date, end_date = date_range.split(" to ")
            return None, start_date, end_date
        elif date_range in period_options:
            return date_range, None, None
        print(Fore.RED + "Invalid input. Here are examples: 2021-01-01 to 2021-12-31, 6mo, 8y" + Style.RESET_ALL)

def get_historical_data(ticker, start_date=None, end_date=None, period=None):
    """
    Fetches historical adjusted close prices for a given ticker.
    """
    if period:
        return yf.download(ticker, period=period)['Adj Close']
    else:
        return yf.download(ticker, start=start_date, end=end_date)['Adj Close']

def calculate_daily_returns(prices):
    """
    Calculates daily returns from price data.
    """
    return prices.pct_change().dropna()

def align_returns(stock_returns, market_returns):
    """
    Aligns stock and market returns by date.
    """
    return pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()

def perform_regression(returns):
    """
    Performs a linear regression of stock returns on market returns.
    """
    X = sm.add_constant(returns['market'])
    return sm.OLS(returns['stock'], X).fit()

def compute_trailing_beta(model):
    """
    Computes the trailing beta and related statistics from the regression model.
    """
    intercept = model.params['const']
    beta = model.params['market']
    p_values = model.pvalues
    confidence_intervals = model.conf_int(alpha=0.05)
    
    return {
        "intercept": intercept,
        "beta": beta,
        "confidence_intervals": confidence_intervals,
        "p_values": p_values
    }

def plot_regression(returns, model, stock_ticker, market_ticker):
    """
    Plots the regression results with confidence intervals.
    """
    intercept = model.params['const']
    beta = model.params['market']
    
    plt.figure(figsize=(16, 9))
    
    # Use scatterplot
    sns.scatterplot(x='market', y='stock', data=returns, color='blue', label='Observations')
    
    # Manually add the regression line
    x = np.linspace(returns['market'].min(), returns['market'].max(), 100)
    y_pred = intercept + beta * x
    plt.plot(x, y_pred, color='orange', label='Regression Line')
    
    se_line = np.sqrt(model.mse_resid) * np.sqrt(1 / len(returns) + (x - returns['market'].mean()) ** 2 / np.sum((returns['market'] - returns['market'].mean()) ** 2))
    ci_upper = y_pred + t.ppf(0.975, df=len(returns) - 2) * se_line
    ci_lower = y_pred - t.ppf(0.975, df=len(returns) - 2) * se_line
    
    plt.plot(x, ci_upper, 'r--', label='95% Confidence Interval')
    plt.plot(x, ci_lower, 'r--')
    
    plt.xlabel('Market Return')
    plt.ylabel('Stock Return')
    plt.title(f'Regression of {stock_ticker} on {market_ticker}')
    plt.legend(loc='upper left')
    
    p_value_regression = model.f_pvalue
    p_value_intercept = model.pvalues['const']
    p_value_beta = model.pvalues['market']
    
    textstr = f"{stock_ticker}\n" \
              f"Statistical test of the regression's (p-value) = {p_value_regression:.3e}\n" \
              f"Beta, slope coefficient = {beta:.4f}\n" \
              f"Statistical test of the beta coef (p-value)= {p_value_beta:.3e}\n" \
              f"Alpha, intercept = {intercept:.4f}\n" \
              f"Statistical test of the intercept (p-value) = {p_value_intercept:.3e}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.show()

if __name__ == "__main__":
    main()
