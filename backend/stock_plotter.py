from typing import List
from typing import Dict


from dataclasses import dataclass, field
import datetime
import time

import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

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


def read_plebdex_as_dataframe(file_path=PLEBDEX_EXCEL_FILE):
    """
    Reads multiple sheets from an Excel file, each sheet named by Year.
    The first row of each sheet is assumed to be column headers (e.g. Ticker, Weight).
    Returns a DataFrame with columns: [Year, Ticker, Weight].
    Performs basic validation:
      - Ensures each sheet name can be parsed as an integer year.
      - Ensures each sheet has columns ["Ticker", "Weight"].
      - Groups by (Year, Ticker) to sum up any duplicate rows.
    """
    logger.debug(f"Reading the plebdex holdings as a dataframe from {file_path}")
    excel_file = pd.ExcelFile(file_path)
    all_sheets = []

    for sheet_name in excel_file.sheet_names:
        # Error-check: ensure sheet_name is convertible to an integer
        year_val = 0
        try:
            year_val = int(sheet_name)
        except ValueError as e:
            raise ValueError(
                f"Sheet name '{sheet_name}' is not a valid integer year. "
                "Please rename or fix the sheet name."
            ) from e

        df_sheet = excel_file.parse(sheet_name, header=0)

        # Inject the parsed year into the DataFrame
        df_sheet["Year"] = year_val

        all_sheets.append(df_sheet)

    # Combine all sheets
    df_plebdex = pd.concat(all_sheets, ignore_index=True)

    return df_plebdex  # has columns Ticker, Weight, Year


def fetch_yearly_stock_prices(year, tickers_list):
    """
    Fetches daily data (Open, High, Low, Close, Volume, etc.)
    for all tickers in tickers_list for the given calendar year.

    Returns a single DataFrame with a MultiIndex or flattened columns
    (depending on group_by setting).
    """
    start_date = datetime.date(year - 1, 12, 1)
    end_date = datetime.date(year + 1, 1, 1)  # up to Jan 1 of next year

    # single call for all tickers if you want bulk fetch
    daily_stock_data = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        group_by="ticker",
        multi_level_index=False,
        threads=False,  # default is true and connection pool limit to yFinance is 10
    )

    # If the result is empty, just return it
    if daily_stock_data.empty:
        return daily_stock_data

    # some of the daily stock data includes empty NaN data
    # let's clean it by filling with the previous day's price
    daily_stock_data.ffill(inplace=True)

    # Flatten columns:
    # Original MultiIndex is like ('SPY','Open'), after flatten => "Open_SPY"
    daily_stock_data.columns = [
        f"{lvl0}_{lvl1}" for lvl0, lvl1 in daily_stock_data.columns
    ]

    # Truncate to just Jan 1–Dec 31 of THIS `year`
    year_start = pd.Timestamp(year, 1, 1)
    year_end = pd.Timestamp(year, 12, 31, 23, 59, 59)  # last second of Dec 31
    daily_stock_data = daily_stock_data.loc[year_start:year_end]

    return daily_stock_data


def capture_all_yearly_data(plebdex):
    """
    Reads each Year from plebdex,
    fetches daily data for the tickers of that year,
    returns a dict {year: DataFrame of daily data}.
    """
    results = {}
    # Group plebdex by year
    for year, df_year in plebdex.groupby("Year"):
        # Gather unique tickers for this year
        tickers_list = df_year["Ticker"].unique().tolist()
        logger.debug(f"Fetching daily data for Year={year}, Tickers={tickers_list}")

        # fetch daily stock price for our tickers from yFinance
        daily_stock_data = fetch_yearly_stock_prices(year, tickers_list)
        results[year] = daily_stock_data

        time.sleep(
            2
        )  # 2-second pause between calls to avoid rate limiting from yFinance

    return results


def find_first_valid_day(df_daily, tickers_list):
    """
    Returns (first_row, first_date) where all "Open_TICKER" columns
    are non-NaN. If none found, return (None, None).
    """
    if df_daily.empty:
        return None, None

    # For each ticker => "SPY_Open", "AAPL_Open", etc.
    open_cols = [f"{ticker}_Open" for ticker in tickers_list]

    # Filter out rows that have NaN in these open columns
    df_valid = df_daily.dropna(subset=open_cols, how="any").copy()

    if df_valid.empty:
        return None, None

    # Sort by date index to ensure the earliest date is first
    df_valid.sort_index(inplace=True)
    first_date = df_valid.index[0]
    first_row = df_valid.iloc[0]
    return first_row, first_date


def compute_portfolio_values(df_plebdex, yearly_data, initial_investment=10_000):
    """
    For each year in 'yearly_data':
      - find the first valid trading day to allocate shares
      - create Value_{ticker} columns = # shares * TICKER_Close
      - sum to PortfolioValue column
      - the last day's PortfolioValue => next year's initial investment
    Returns:
      portfolio_history: {year: df_daily (with Value_ cols, PortfolioValue col)}
      final_investment_value: float
    """
    portfolio_history = {}
    investment_value = initial_investment

    # sort the years so we process them in ascending order
    for year in sorted(yearly_data.keys()):
        df_daily = yearly_data[year].copy()

        # subset of plebdex for this year (tickers + weights)
        df_year = df_plebdex[df_plebdex["Year"] == year]
        if df_daily.empty or df_year.empty:
            logger.debug(f"Year={year} has no data or no tickers, skipping.")
            portfolio_history[year] = df_daily
            continue

        # ========== Step A: Find first valid day for opening prices ==========
        tickers_list = df_year["Ticker"].unique().tolist()
        first_row, first_date = find_first_valid_day(df_daily, tickers_list)
        if first_row is None:
            logger.debug(f"No valid first trading day for Year={year}, skipping.")
            portfolio_history[year] = df_daily
            continue

        # ========== Step B: Allocate shares based on initial investment, ==========
        # ========== weights, open price on first_date ==========
        num_shares = {}
        for _, row in df_year.iterrows():
            ticker = row["Ticker"]
            weight = row["Weight"]
            open_col = f"{ticker}_Open"
            if open_col not in first_row:
                logger.debug(f"Warning: {open_col} not in first_row for {ticker}")
                continue

            open_price = first_row[open_col]
            if pd.isna(open_price) or open_price <= 0:
                logger.debug(f"Warning: Invalid open_price for {ticker} in Year={year}")
                continue

            allocated_dollars = investment_value * weight
            shares = allocated_dollars / open_price
            num_shares[ticker] = shares

        # ========== Step C: Compute daily Value_{ticker} and PortfolioValue ==========
        for ticker, shares in num_shares.items():
            close_col = f"{ticker}_Close"
            if close_col in df_daily.columns:
                df_daily[f"Value_{ticker}"] = df_daily[close_col] * shares
            else:
                logger.debug(
                    f"Warning: {close_col} missing in df_daily for Year={year}, Ticker={ticker}"  # noqa: E501
                )
                df_daily[f"Value_{ticker}"] = 0.0

        # Sum all Value_{ticker} columns => PortfolioValue
        value_cols = [c for c in df_daily.columns if c.startswith("Value_")]
        df_daily["PortfolioValue"] = df_daily[value_cols].sum(axis=1)

        # ========== Step D: Determine final day's portfolio value = next year's investment ========== # noqa: E501
        # first, drop rows that might be NaN in "PortfolioValue" if necessary
        df_valid_close = df_daily.dropna(subset=["PortfolioValue"])
        if df_valid_close.empty:
            final_value = 0.0
        else:
            # take the last row as your final day
            last_day_date = df_valid_close.index[-1]
            final_value = df_valid_close.loc[last_day_date, "PortfolioValue"]

        logger.debug(
            f"Year={year} => first valid day={first_date}, final day={last_day_date}, final portfolio=${final_value:,.2f}"  # noqa: E501, E231
        )
        investment_value = final_value

        # store the augmented DataFrame
        portfolio_history[year] = df_daily

    return portfolio_history, investment_value


def plot_plebdex_vs_benchmarks(
    portfolio_history,
    bench_tickers=["SPY", "QQQ"],
    label_plebdex="Plebdex",
    plot_title="Plebdex vs Benchmarks (Normalized)",
):
    """
    Merges the daily portfolio DataFrames from 'portfolio_history' into one,
    finds a valid 'start' day for normalization, fetches & normalizes benchmarks,
    then plots them all together.

    :param portfolio_history: dict { year: df_daily_with_PortfolioValue }
    :param bench_tickers: list of benchmark ticker symbols for comparison
    :param label_plebdex: the label to give your plebdex line in the legend
    :param plot_title: chart title
    """
    # -----------------------------
    # STEP A: Combine all years' portfolio data into one DataFrame
    # -----------------------------
    df_pleb_all = pd.concat(
        [
            portfolio_history[y][["PortfolioValue"]]
            for y in sorted(portfolio_history.keys())
        ],
        axis=0,
    )
    df_pleb_all.sort_index(inplace=True)

    # Drop rows where PortfolioValue is NaN or zero
    df_pleb_all = df_pleb_all.dropna(subset=["PortfolioValue"])
    df_pleb_all = df_pleb_all[df_pleb_all["PortfolioValue"] > 0]

    if df_pleb_all.empty:
        logger.debug("No valid portfolio data to plot.")
        return

    # 1) Identify the first valid row to normalize
    first_valid_date = df_pleb_all.index[0]
    first_val = df_pleb_all.loc[first_valid_date, "PortfolioValue"]

    # 2) Create a normalized column
    df_pleb_all[label_plebdex] = df_pleb_all["PortfolioValue"] / first_val

    # -----------------------------
    # STEP B: Fetch benchmark data (covering the date range of the plebdex)
    # -----------------------------
    # We'll fetch from the earliest to a bit past the latest date in df_pleb_all
    start_date = df_pleb_all.index.min().date()
    end_date = df_pleb_all.index.max().date() + datetime.timedelta(days=1)

    logger.debug(f"Fetching benchmarks {bench_tickers} from {start_date} to {end_date}")
    df_bench = yf.download(
        bench_tickers, start=start_date, end=end_date, group_by="ticker", threads=False
    )

    if not df_bench.empty:
        # Flatten columns
        df_bench.columns = [f"{c[0]}_{c[1]}" for c in df_bench.columns]
        # Forward fill missing
        df_bench.ffill(inplace=True)

        # -----------------------------
        # STEP C: Create normalized columns for each benchmark
        # -----------------------------
        for ticker in bench_tickers:
            close_col = f"{ticker}_Close"
            if close_col not in df_bench.columns:
                logger.debug(
                    f"Warning: {close_col} not in df_bench; skipping {ticker}"  # noqa: E501, E702
                )
                continue

            # Drop rows where the close is zero or NaN
            df_bench = df_bench.dropna(subset=[close_col])
            df_bench = df_bench[df_bench[close_col] > 0]

            if df_bench.empty:
                continue

            # Normalized to the *first valid* close
            first_bench_val = df_bench[close_col].iloc[0]
            norm_col = f"{ticker}_Norm"
            df_bench[norm_col] = df_bench[close_col] / first_bench_val
    else:
        logger.debug(
            "Benchmark data is empty or failed to fetch. Plotting only Plebdex."
        )

    # -----------------------------
    # STEP D: Merge for plotting
    # -----------------------------
    # Keep only the columns we care about
    if not df_bench.empty:
        norm_cols = [
            f"{t}_Norm" for t in bench_tickers if f"{t}_Norm" in df_bench.columns
        ]
        df_plot = pd.merge(
            df_pleb_all[[label_plebdex]],
            df_bench[norm_cols],
            how="outer",
            left_index=True,
            right_index=True,
        )
    else:
        df_plot = df_pleb_all[[label_plebdex]]

    df_plot.sort_index(inplace=True)

    if df_plot.empty:
        logger.debug("Merged DataFrame is empty, nothing to plot.")
        return

    logger.debug("=== First rows of df_pleb_all ===")
    logger.debug(df_pleb_all.head(20))

    # -----------------------------
    # STEP E: Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))
    # Plot the plebdex line
    df_plot[label_plebdex].plot(label=label_plebdex)

    # Plot each benchmark line
    if not df_bench.empty:
        for ticker in bench_tickers:
            norm_col = f"{ticker}_Norm"
            if norm_col in df_plot.columns:
                df_plot[norm_col].plot(label=ticker)

    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # initialize our logger
    initialize_logger()

    # read the plebdex holdings from excel, returning a data structure
    # containing all holdings by year
    # plebdex = read_plebdex_and_generate_holdings()

    plebdex = read_plebdex_as_dataframe()
    logger.info(plebdex)

    # 2. Fetch daily data for each year
    yearly_data = capture_all_yearly_data(plebdex)
    logger.info(yearly_data)
    # yearly_data[2023] -> daily DataFrame for tickers in 2023,
    # columns like "Open_SPY", "Close_SPY", ...

    # 3. Compute daily portfolio & get final value
    portfolio_history, final_value = compute_portfolio_values(plebdex, yearly_data)

    # 4. combine all years' portfolio data and plot it
    plot_plebdex_vs_benchmarks(
        portfolio_history,
        bench_tickers=["SPY", "QQQ"],  # or any list of tickers
        label_plebdex="Plebdex",
        plot_title="Plebdex vs SPY vs QQQ (Normalized)",
    )

    logger.info(
        f"Final portfolio value after last year: ${final_value:,.2f}"  # noqa: E231
    )


if __name__ == "__main__":
    main()
