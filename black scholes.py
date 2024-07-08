#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alexandra
"""

# Black Scholes and Binomial Option Pricing
# Analyzing Apple (AAPL) stock from 2016-2021, then looking to see the call option price
# Using Black Scholes to approximate the call option price, and binomial to accurately predict stock prices

import numpy as np
import pandas as pd
import math
from scipy.stats import norm 
import matplotlib.pyplot as plt

def BlackScholes_call_price(S, K, T, r, sigma):
    # Variable definitions
    # sigma = volatility of the stock
    
    # Calculating d1 and d2
    d1 = (np.log(S/K) + (r + ((sigma)**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    callprice = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)

    return callprice


def Binomial_call_price(S, K, t, r, sigma, n, dataframe): # Credit in paper
    # Transitioning from Black Scholes to Binomial using parameters in powerpoint
    deltat = T/n 
    R = np.exp(r*deltat) # Risk free interest rate
    U = np.exp(sigma*(np.sqrt(deltat))) # Increase factor
    D = np.exp(-(sigma*(np.sqrt(deltat)))) # Decrease factor
    p = (R-D)/(U-D) # Risk neutral factor
    print(deltat, R, U, D, p) 
    
    # Start with the last node payoff
    optionprice = max(0, S * U**n - K)
    
    for i in range(n - 1, -1, -1):
        S *= D / U  # Calculate the stock price at this node
        optionprice = max(optionprice, max(0, S - K) * np.exp(-r * deltat))  # Continuation value
    
    return optionprice
    
def binomial_tree(S, K, T, r, sigma, n):
    deltat = T / n 
    R = np.exp(r * deltat)  # Risk-free interest rate
    U = np.exp(sigma * np.sqrt(deltat))  # Upward movement
    D = np.exp(-sigma * np.sqrt(deltat))  # Downward movement

    # Initializing stock price tree
    stockprice = np.zeros((n + 1, n + 1))
    stockprice[0, 0] = S

    # Generate stock price tree
    for i in range(1, n + 1):
        stockprice[i, 0] = stockprice[i - 1, 0] * U
        for j in range(1, i + 1):
            stockprice[i, j] = stockprice[i - 1, j - 1] * D

    # Initialize option price tree
    optionprice = np.zeros((n + 1, n + 1))

    # Calculate option prices at expiration (time T)
    for j in range(n + 1):
        optionprice[n, j] = max(0, stockprice[n, j] - K)

    # Backward induction to compute option prices at earlier nodes
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optionprice[i, j] = max(0, stockprice[i, j] - K, np.exp(-r * deltat) * (R * optionprice[i + 1, j] + (1 - R) * optionprice[i + 1, j + 1]))

    return optionprice


def plot_binomial_tree(optionprice):
    n = len(optionprice)
    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(n):
        for j in range(i + 1):
            plt.plot(j - i / 2, n - i, 'o', color='black', markersize = 3)  # Plot nodes as circles
            plt.text(j - (i+0.5) / 2, n - i, f'{option_price[i, j]:.2f}', ha='center', va='center', color='white')  # Show option prices

            # CCreating lines on binomial plot
            if i < n - 1:
                plt.plot([j - i / 2, j - i / 2 + 0.5], [n - i, n - i - 1], color='black')  # Connect to nodes below
                plt.plot([j - i / 2, j - i / 2 - 0.5], [n - i, n - i - 1], color='black')  # Connect to nodes below
                
                arrowlength = 0.3
                # Adding arrows to show the direction we are moving in
                plt.annotate('', xy = (j - 1 / 2 + 0.5, n), xytext = (j - i / 2, n - i), arrowprops = dict(arrowstyle='->', color='green'))  # up arrow
                plt.annotate('', xy = (j - i / 2 - 0.5, n - i - 1), xytext = (j - i / 2, n - i), arrowprops = dict(arrowstyle='->', color='red'))  # down arrow

    plt.title("Binomial Tree for Apple Option Pricing")
    plt.xlabel("Stock Price")
    plt.ylabel("Time Steps")
    plt.grid(False)
    plt.show()
    
# Program
# Opening file and doing all that
file = '/Users/alexandra/Desktop/MAP4103 Finance Project/aapl_raw_data_120623.csv'
initial_df = pd.read_csv(file)

# Converting date column to datetime format
initial_df['date'] = pd.to_datetime(initial_df['date'])

start_date = '2016-01-01'
end_date = '2021-12-31'

# Filter the DataFrame based on the date range
df = initial_df[(initial_df['date'] >= start_date) & (initial_df['date'] <= end_date)]

print(df)

# Calculate daily returns
df['Daily Returns'] = df['adjusted_close'].pct_change()

print("daily returns column:\nrow number\tpercent change\n", df['Daily Returns'])

# Calculate historical volatility (standard deviation of returns over a certain period)
historical_volatility = df['Daily Returns'].rolling(window=20).std().dropna().mean().round(4)  # Adjust window size as needed
print(historical_volatility) # good

# Extracting most recent 2021 price to calculate certain variables in Black Scholes
# Finding S and K
data_2021 = df[df['date'].dt.year == 2021]
data_2021_sorted = data_2021.sort_values(by = 'date', ascending = False)
recent_close_price_2021 = data_2021_sorted.iloc[0]['close'] # good, this is S and K
K = 175.56 # from data set i dont feel like getting it out

# Finding T (time to maturity of stock)
expiration = pd.to_datetime('2022-01-01')
df['time to maturity'] = (expiration - df['date']).dt.days/365.25
print(df[['date', 'time to maturity']])

# or just set it = to 1
T = 1

# Federal reserve to calculate risk free interest rates (r)
# say its 0.05 like in sample for now
risk_free_interest = 0.035

# Black Scholes estimate
callprice1 = BlackScholes_call_price(recent_close_price_2021, recent_close_price_2021, T, risk_free_interest, historical_volatility)
print('black scholes approximation:', callprice1) # good

# Binomial Call Price
n = 5

callprice2 = Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, n, df)
print('binomial call price: ', callprice2)
print(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, n)
option_price = binomial_tree(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, n)  # Calculate option prices using binomial tree
plot_binomial_tree(option_price)


xvals = [3, 5, 10, 25, 50, 75, 100]

yvals = [Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 3, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 5, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 10, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 25, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 50, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 75, df),
                             Binomial_call_price(recent_close_price_2021, K, T, risk_free_interest, historical_volatility, 100, df)]
print(yvals)

plt.plot(xvals, yvals, marker = 'o')
plt.show()

plt.plot(xvals, np.log(yvals), marker = 'o', color = 'green')
plt.show()

# Option price proportion
for i in yvals:
    result = (i / recent_close_price_2021) * 100
    print(result)





