import datetime
import time

import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

import logging as log

logger = log.getLogger(__name__)
DEFAULT_LOG_LEVEL = log.INFO
DEFAULT_LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

INITIAL_INVESTMENT_DOLLARS = 10_000
PLEBDEX_EXCEL_FILE = "./data/plebdex-weights.xlsx"


def initialize_logger():
    """
    This function initializes our application logger's format and log level
    """
    log.basicConfig(format=DEFAULT_LOG_FORMAT, level=DEFAULT_LOG_LEVEL)


def make_benchmark_plebdex(ticker, first_year, last_year):
    """
    Creates a DataFrame with one row per year from first_year to last_year,
    setting Ticker = <ticker> and Weight = 1.0 (100% in that ticker).
    """
    rows = []
    for y in range(first_year, last_year + 1):
        rows.append({"Year": y, "Ticker": ticker, "Weight": 1.0})
    df_bench = pd.DataFrame(rows, columns=["Year", "Ticker", "Weight"])
    return df_bench


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
        # multi_level_index=False,
        threads=False,  # default is true and connection pool limit to yFinance is 10
        actions=True,  # fetches dividend information
    )

    # If the result is empty, just return it
    if daily_stock_data.empty:
        return daily_stock_data

    if isinstance(daily_stock_data.columns, pd.MultiIndex):
        daily_stock_data.columns = [
            f"{lvl0}_{lvl1}" for (lvl0, lvl1) in daily_stock_data.columns
        ]

    # some of the daily stock data includes empty NaN data
    # let's clean it by filling with the previous day's price
    daily_stock_data.ffill(inplace=True)

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


def shift_dividends_by_offset(df_daily, tickers_list, offset=30):
    """
    For each ticker, we shift the ex-dividend data (Ticker_Dividends)
    30 days forward to approximate the actual pay date.
    Returns a new DataFrame that includes Ticker_PaidDividends columns.
    """
    df_result = df_daily.copy()

    for ticker in tickers_list:
        ex_col = f"{ticker}_Dividends"
        pay_col = f"{ticker}_PaidDividends"

        if ex_col not in df_result.columns:
            # no dividends column for this ticker
            continue

        # Original ex-date series
        ex_series = df_result[ex_col]

        # Create a shifted copy by 30 days
        shifted = ex_series.copy()
        shifted.index = shifted.index + pd.DateOffset(days=offset)

        # Reindex onto df_result's index, fill missing with 0.0
        shifted = shifted.reindex(df_result.index, fill_value=0.0)

        # Put this in a new column "Ticker_PaidDividends"
        df_result[pay_col] = shifted

    return df_result


def compute_portfolio_values(
    df_plebdex, yearly_data, initial_investment=INITIAL_INVESTMENT_DOLLARS
):
    """
    For each year in 'yearly_data':
      - find the first valid trading day to allocate shares
      - create Value_{ticker} columns = # shares * TICKER_Close
      - sum to PortfolioValue column
      - the last day's PortfolioValue => next year's initial investment
    Incorporates daily dividend reinvestment:
      - Each day, if "Ticker_Dividends" > 0, we buy additional shares of Ticker
        at that day's Close price.
      - The final day value => next year's initial investment.
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

        tickers_list = df_year["Ticker"].unique().tolist()

        # Shift ex-div data by 30 days to approximate pay date
        df_daily = shift_dividends_by_offset(df_daily, tickers_list)

        # ========== Step A: Find first valid day for opening prices ==========
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

        logger.debug(f"=== End of Year {year} Holdings ===")
        for t, shares in num_shares.items():
            logger.debug(f"    {t}: {shares:.2f} shares")  # noqa: E231
        investment_value = final_value

        # store the augmented DataFrame
        portfolio_history[year] = df_daily

    return portfolio_history, investment_value


def run_plebdex_portfolio(df_plebdex, initial_investment=INITIAL_INVESTMENT_DOLLARS):
    """
    High-level function that:
      1) Captures all yearly data from Yahoo (bulk fetching each year's tickers)
      2) Computes the daily portfolio, returning the final value.
    Returns:
      portfolio_history: {year: DataFrame with daily PortfolioValue, etc.}
      final_value: float final portfolio after the last year
    """
    # Step A: Capture the yearly data
    yearly_data = capture_all_yearly_data(df_plebdex)

    # Step B: Compute the portfolio
    portfolio_history, final_val = compute_portfolio_values(
        df_plebdex, yearly_data, initial_investment=initial_investment
    )

    return portfolio_history, final_val


def plot_portfolios(portfolio_histories, plot_title="Comparison"):
    """
    Takes a dictionary of label -> portfolio_history,
    where each portfolio_history is {year: df_daily} with a 'PortfolioValue' column.
    Merges them all, normalizes each to start at 1.0, and plots in one chart.
    """
    if not portfolio_histories:
        log.warning("No portfolios to plot.")
        return

    # We'll create an empty DataFrame that we'll merge each label's data into
    master_df = pd.DataFrame()

    # For each label (e.g. "Plebdex", "SPY", "QQQ"), combine its years into one DF
    # then normalize the first valid day to 1.0, and merge into master_df
    for label, hist_dict in portfolio_histories.items():
        # 1) Combine all years
        df_port = pd.concat(
            [hist_dict[y][["PortfolioValue"]] for y in sorted(hist_dict.keys())], axis=0
        ).sort_index()

        # 2) Drop invalid
        df_port.dropna(subset=["PortfolioValue"], inplace=True)
        df_port = df_port[df_port["PortfolioValue"] > 0]

        if df_port.empty:
            log.warning(f"No valid data for {label}, skipping.")
            continue

        # 3) Normalize
        first_val = df_port["PortfolioValue"].iloc[0]
        df_port[label] = df_port["PortfolioValue"] / first_val

        # 4) We'll merge df_port[label] into master_df
        df_port_to_merge = df_port[[label]]  # just the normalized column

        # Outer merge on index
        if master_df.empty:
            master_df = df_port_to_merge
        else:
            master_df = pd.merge(
                master_df,
                df_port_to_merge,
                how="outer",
                left_index=True,
                right_index=True,
            )

    if master_df.empty:
        log.warning("No merged data to plot after processing all portfolios.")
        return

    # 5) Plot each label's line
    plt.figure(figsize=(10, 6))
    for label in portfolio_histories.keys():
        if label in master_df.columns:
            master_df[label].plot(label=label)

    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # initialize our logger
    initialize_logger()

    # 1) Read plebdex
    df_plebdex = read_plebdex_as_dataframe(PLEBDEX_EXCEL_FILE)
    # 2) Run pipeline for main plebdex
    portfolio_history_pleb, final_val_pleb = run_plebdex_portfolio(
        df_plebdex, INITIAL_INVESTMENT_DOLLARS
    )

    # 3) Single-ticker "fake" plebdex for SPY
    df_spy_plebdex = make_benchmark_plebdex("SPY", 2023, 2025)  # define the date range
    portfolio_history_spy, final_val_spy = run_plebdex_portfolio(
        df_spy_plebdex, INITIAL_INVESTMENT_DOLLARS
    )

    # 4) Another benchmark: QQQ
    df_qqq_plebdex = make_benchmark_plebdex("QQQ", 2023, 2025)
    portfolio_history_qqq, final_val_qqq = run_plebdex_portfolio(
        df_qqq_plebdex, INITIAL_INVESTMENT_DOLLARS
    )

    # 5) Prepare the dict for plotting
    portfolios_to_plot = {
        "Plebdex": portfolio_history_pleb,
        "SPY": portfolio_history_spy,
        "QQQ": portfolio_history_qqq,
    }

    # 6) Print final returns if needed
    logger.info(f"Plebdex ended with ${final_val_pleb:.2f}")  # noqa: E231
    logger.info(f"SPY ended with ${final_val_spy:.2f}")  # noqa: E231
    logger.info(f"QQQ ended with ${final_val_qqq:.2f}")  # noqa: E231

    # 7) Plot them, no direct yfinance calls inside the plot function
    plot_portfolios(portfolios_to_plot, "Plebdex vs. Benchmarks (Normalized)")


if __name__ == "__main__":
    main()
