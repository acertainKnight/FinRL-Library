"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta



class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        i = 0
        d_list = self.date_list()
        for tic in self.ticker_list:
            # if i == 50:
            #     break
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            if len(temp_df) < 1:
                continue
            # elif temp_df.index[0] not in d_list:
                # date_df = pd.DataFrame({'date_y': pd.date_range(start=self.start_date,
                #                                            end=self.end_date,
                #                                            freq='D').to_list()})
                # date_df = pd.DataFrame({'date_y': date_col.to_list()})
                # temp_date_df = pd.merge(date_df, temp_df, how='left', left_on='date_y', right_index=True)
                # temp_date_df = temp_date_df.fillna(0)
                # temp_date_df.drop('date_y', axis=1)
                # print('Tic not available over period; {}'.format(tic))
                # continue
            else:
                # date_col = temp_df.index
                temp_date_df = temp_df.copy()
            temp_date_df["tic"] = tic
            # yfticker = yf.Ticker(tic)
            # temp_date_df["sector"] = yfticker.info['sector']
            # temp_date_df.sector = pd.Categorical(temp_date_df.sector)
            # temp_date_df['sector'] = temp_date_df.sector.cat.codes
            temp_date_df.drop_duplicates(inplace=True)
            data_df = data_df.append(temp_date_df)
            i += 1
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        print(list(data_df))
        # data_df.drop('index', axis=1, inplace=True)
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
                # "sector"
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df[data_df['day'] < 5]
        # drop missing data
#         data_df = data_df.fillna(-1)
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)
        data_df.drop_duplicates(inplace=True)
        data_df.to_csv('dow30.csv')
        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

    def date_list(self):
        s = self.start_date
        date = datetime.strptime(s, "%Y-%m-%d")
        dates = []
        for i in range(1, 3):
            dates.append(date + timedelta(days=i))
        return dates
