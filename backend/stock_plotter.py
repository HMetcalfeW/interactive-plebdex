from typing import List
from typing import Dict


from dataclasses import dataclass, field
import datetime

import pandas as pd

import yfinance as yf

# import matplotlib.pyplot as plt


import logging as log

logger = log.getLogger(__name__)
DEFAULT_LOG_LEVEL = log.INFO
DEFAULT_LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

INITIAL_INVESTMENT_DOLLARS = 10000
PLEBDEX_EXCEL_FILE = "./data/plebdex-weights.xlsx"


@dataclass
class Holding:
    """
    This represents an investable security holding
    """

    ticker: str = ""
    initial_weight: float = 0.0
    num_of_shares: float = 0.0


@dataclass
class AssetValue:
    """
    A dataclass representing an asset's value used to plot the
    value over time
    """

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

    annual_return: float = 0.0
    holdings: List[Holding] = field(default_factory=list)
    daily_asset_values: Dict[datetime.date, AssetValue] = field(default_factory=dict)


def initialize_logger():
    """
    This function initializes our application logger's format and log level
    """
    log.basicConfig(format=DEFAULT_LOG_FORMAT, level=DEFAULT_LOG_LEVEL)


def read_plebdex_and_generate_holdings(file_path=PLEBDEX_EXCEL_FILE):
    """
    This function reads in the plebdex from excel
    The excel file contains multiple sheets, each representing one year
    """
    logger.debug("Reading the plebdex holdings from " + file_path)

    excel_file = pd.ExcelFile(file_path)
    logger.debug("The contents of the plebdex excelsheet " + str(excel_file))

    holdings_by_year = {}

    for sheet_name in excel_file.sheet_names:
        # Convert the sheet name to an integer year (if it's a string like "2020")
        year_val = int(sheet_name)

        # Parse the sheet
        df = excel_file.parse(sheet_name)

        holdings = []

        # Assume df has columns: "Ticker" (str), "Weight" (float)
        for row in df.itertuples(index=False):
            # Access attributes exactly matching column names: row.Ticker, row.Weight
            holding = Holding(ticker=row.Ticker, initial_weight=row.Weight)
            holdings.append(holding)

        holdings_by_year[year_val] = holdings

    return holdings_by_year


def normalize(df):
    """
    Normalizes the first value of each column to 1,
    and scales the subsequent values accordingly.
    """
    return df / df.iloc[0]


def fetch_stock_data(year, ticker):
    """
    Fetches historical data for a given ticker from Yahoo Finance.
    Returns a DataFrame with the 'Close' price for a given holding.
    """

    # Given the trading year, calculate our start date of Jan 1
    start_date = datetime.date(year, 1, 1)

    # Set the upper bound to
    end_date = datetime.date(year + 1, 1, 1)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    logger.debug(stock_data)

    return stock_data


def rebalance():
    log.info("TODO")


def calculate_annual_return(plebdex):
    """
    Calculates the annual return for the plebdex.

    This calculation is done by taking the initial dollar investment,
    and calculating the number of owned shares of an asset.
    At the end of each year, the portfolio is rebalanced into the
    subsequent years holdings
    """

    # calculate the number of shares for each holding

    # calculate the daily close of the plebdex to build data points

    # each subsequent year rebalance the total portfolio value into the new holdings

    # calculate the daily close of the plebdex to build data points

    investment_value = INITIAL_INVESTMENT_DOLLARS

    for year, holdings in plebdex.items():
        year_str = str(year)

        logger.info("Started " + year_str + " with $" + str(investment_value))

        # for each holding fetch the stock data for that year
        for holding in holdings:
            stock_data = fetch_stock_data(year, holding.ticker)
            logger.debug(stock_data)

            if stock_data.empty:
                logger.info("No stock data for" + holding.ticker)
                continue

            # get the Jan 1 price opening price to calculate our initial share count
            jan_1_price_dollars = stock_data.iloc[0]["Open"].values[0]
            logger.debug(
                "January " + year_str + " Opening Price " + str(jan_1_price_dollars)
            )

            # calculate the initial number of holdings for the year
            holding.num_of_shares = (
                investment_value * holding.initial_weight
            ) / jan_1_price_dollars
            logger.debug("Holding " + str(holding))

            # todo: figure out the appropriate calculations for dividends

            # get the last closing price
            end_of_year_price_dollars = stock_data.iloc[len(stock_data) - 1][
                "Close"
            ].values[0]
            logger.debug(end_of_year_price_dollars)

            # calculate the total asset return based on the number of shares held
            investment_value = investment_value + (
                (end_of_year_price_dollars - jan_1_price_dollars)
                * holding.num_of_shares
            )

        logger.info("Ended " + year_str + " with $" + str(investment_value))


def main():
    # initialize our logger
    initialize_logger()

    # read the plebdex holdings from excel, returning a data structure
    # containing all holdings by year
    plebdex = read_plebdex_and_generate_holdings()

    # calculate the rolling sum of returns
    calculate_annual_return(plebdex)

    # normalize the plebdex

    # plot the data points

    # fetch_stock_data("SPY", 2023)

    # fetch our holdings from the plebdex
    logger.info(plebdex)

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
