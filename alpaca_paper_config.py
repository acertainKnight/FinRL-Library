import os

ALPACA_ENDPOINT = "https://api.alpaca.markets"
ALPACA_KEYID = "AKFIW68Z45DJVQHJJP5L"
ALPACA_SECRETKEY = "FtUVTAl9PzUaKwSf3XMPOimUk8UAnLiSuqkTO5CD"

os.environ["APCA_API_KEY_ID"] = ALPACA_KEYID
os.environ["APCA_API_SECRET_KEY"] = ALPACA_SECRETKEY
os.environ["APCA_API_BASE_URL"] = ALPACA_ENDPOINT

"""
Download historical minute by minute data for a symbol between a range of dates
"""

# Install a pip package in the currenta Jupyter kernel
import os

import pandas as pd
from tqdm import tqdm

import alpaca_trade_api as tradeapi
from finrl.config import config # loads API Keys as environment variables


def main():
    api = tradeapi.REST()

    DATADIR = os.path.join(os.getcwd(),'datasets/ALPACA', 'SP500') # download directory for the data
    SYMBOLS = config.SP_500_TICKER# list of symbols we're interested
    FROM_DATE = '2009-01-01'
    TO_DATE = '2021-01-01'

    # create data directory if it doesn't exist
    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)

    date_range = pd.date_range(FROM_DATE, TO_DATE)
    for symbol in SYMBOLS:
        symbol_df = pd.DataFrame()
        fname = f'{symbol}-{FROM_DATE}.csv'  # for example, AAPL-2016-01-04.csv
        full_name = os.path.join(DATADIR, fname)
        if os.path.exists(full_name):
            # data file already exists, not necessary to download
            continue
        for fromdate in tqdm(date_range):
            if fromdate.dayofweek > 4:
                # it's a weekend
                continue

            _from = fromdate.strftime('%Y-%m-%d')
            _to = (fromdate + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # download data as a pandas dataframe format
            df = api.polygon.historic_agg_v2(
                symbol=symbol,
                multiplier=1,
                timespan='minute',
                _from=_from,
                to=_to,
                unadjusted=False
            ).df

            if df.empty:
                tqdm.write(f'Error downloading data for date {symbol} {_from}')
                continue

            # filter times in which the market in open
            # df = df.between_time('9:30', '16:00')

            # saving csv for the data of the day in DATADIR/fname
            symbol_df = symbol_df.append(df)

        symbol_df.to_csv(full_name)

if __name__ == "__main__":
    main()
