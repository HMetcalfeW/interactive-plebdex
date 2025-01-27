from typing import List
from typing import Dict


from dataclasses import dataclass
import datetime

import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt


import logging as log

logger = log.getLogger(__name__)
DEFAULT_LOG_LEVEL = log.INFO
DEFAULT_LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

STARTING_DOLLARS = 100
PLEBDEX_EXCEL_FILE = "./data/plebdex-weights.xlsx"


@dataclass
class Holding:
    """
    This represents an investable security holding
    """

    symbol: str
    initial_weight: float
    number_of_shares: float


@dataclass
class AssetValue:
    open: float = 0.0
    close: float = 0.0
    high: float = 0.0
    low: float = 0.0


@dataclass
class plebdex:
    """
    This a portfolio and its holdings for a given year
    along with a calculated daily asset value using the portfolio's holdings
    """

    holdings: List[Holding]
    daily_asset_values: Dict[datetime.date, AssetValue]


def initialize_logger():
    """
    This function initializes our application logger's format and log level
    """
    log.basicConfig(format=DEFAULT_LOG_FORMAT, level=DEFAULT_LOG_LEVEL)


def read_plebdex_holdings(file_path=PLEBDEX_EXCEL_FILE):
    logger.info(file_path)

    excel_sheet = pd.read_excel(file_path)
    logger.info(excel_sheet)


# def fetch_stock_data(ticker, start_date, end_date):
#     """
#     Fetches historical data for a given ticker from Yahoo Finance.
#     Returns a DataFrame with the 'Adj Close' price.
#     """
#     data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
#     print("Downloaded columns:", data.columns)  # Debug print
#     # Keep only the adjusted close column
#     return data[['Close']].rename(columns={'Close': ticker})


def normalize(df):
    """
    Normalizes the first value of each column to 1,
    and scales the subsequent values accordingly.
    """
    return df / df.iloc[0]


def fetch_stock_data(ticker, year):
    """
    Fetches historical data for a given ticker from Yahoo Finance.
    Returns a DataFrame with the 'Close' price for a given holding.
    """
    # Given the trading year, calculate our start date of Jan 1
    start_date = datetime.date(year, 1, 1)

    # Set the upper bound to
    end_date = datetime.date(year + 1, 1, 1)
    stock_data = yf.download("SPY", start=start_date, end=end_date)

    if not ("Close" in stock_data.columns):
        logger.debug("Close is not in the data pulled from yFinance")
        return

    stock_data = stock_data[["Close"]].rename(columns={"Close": ticker})
    logger.info(stock_data)


def main():
    # initialize our logger
    initialize_logger()

    # read the plebdex holdings from excel, returning a data structure
    # containing all holdings by year

    # calculate the number of shares for each holding given the STARTING_DOLLARS for the first year

    # calculate the daily close of the plebdex to build data points

    # each subsequent year rebalance the total portfolio value into the new holdings

    # calculate the daily close of the plebdex to build data points

    # normalize the plebdex

    # plot the data points

    fetch_stock_data("SPY", 2023)
    read_plebdex_holdings()
    # loadPlebdex()
    # Input parameters
    # my_ticker = "AAPL"

    # benchmark_ticker = "SPY"
    # start_date = "2020-01-01"
    # end_date   = "2025-01-25"

    # # Fetch data for my ticker
    # my_data = fetch_stock_data(my_ticker, start_date, end_date)

    # # Fetch data for benchmark
    # bench_data = fetch_stock_data(benchmark_ticker, start_date, end_date)

    # # Combine into one DataFrame for easy comparison
    # combined_df = pd.concat([my_data, bench_data], axis=1).dropna()

    # # Normalize so both series start at 1.0
    # normalized_df = normalize(combined_df)

    # # Plot
    # plt.figure(figsize=(10, 6))
    # for col in normalized_df.columns:
    #     plt.plot(normalized_df.index, normalized_df[col], label=col)

    # plt.title(f"Performance of {my_ticker} vs {benchmark_ticker}")
    # plt.xlabel("Date")
    # plt.ylabel("Normalized Price")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
