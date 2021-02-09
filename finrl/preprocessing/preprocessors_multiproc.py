import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from finrl.config import config
from fredapi import Fred
import pandas_market_calendars as mcal
from multiprocessing import get_context, cpu_count


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        tempindex_list = list(range(0, len(df.tic.unique()), int(len(df.tic.unique())/cpu_count())))
        print(len(df.tic.unique()))
        print(tempindex_list)
        index_list = []
        for i in range(len(tempindex_list)):
            print(i)
            if i == list(range(len(tempindex_list)))[-1]:
                _ = [tempindex_list[i], len(df.tic.unique())]
            else:
                _ = [tempindex_list[i], tempindex_list[i+1]]
            index_list.append(_)

        if self.use_technical_indicator == True:
            with get_context("spawn").Pool() as pool:
                result = pool.map(self.add_technical_indicator(), index_list)
            # add technical indicators using stockstats
            temp = pd.DataFrame()
            for res in result:
                temp.append(res)
            df = temp
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence == True:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature == True:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")
#         df = df.shift(1)
        df = self.standardize_dates(df)
        # fill the missing values at the beginning and the end
        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill").fillna(method="ffill")
        df.drop_duplicates(inplace=True)
        # list_ticker = df["tic"].unique().tolist()
        # list_date = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))
        # combination = list(df.product(list_date, list_ticker))
        #
        # processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"],
        #                                                                           how="left")
        # processed_full = processed_full[processed_full['date'].isin(df['date'])]
        # processed_full = processed_full.sort_values(['date', 'tic'])
        #
        # processed_full = processed_full.fillna(0)

        print("Shape of DataFrame (w/indicators): ", df.shape)
        return df

    def add_technical_indicator(self, data):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df[indicator] = indicator_df
        return df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        unique_ticker = df.tic.unique()
        daily_return_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = df[df.tic == unique_ticker[i]].close.pct_change(1)
                temp_indicator = pd.DataFrame(temp_indicator)
                daily_return_df = daily_return_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                print(e)

        df["daily_return"] = daily_return_df.values

        df['log_volume'] = np.log(df.volume * df.close)
        df['change'] = np.divide(np.subtract(df.close.values, df.open.values), df.close.values)
        df['daily_variance'] = np.divide(np.subtract(df.high.values, df.low.values), df.close.values)
        df['close_boll_ub'] = np.subtract(df.boll_ub.values, df.close.values)
        df['close_boll_lb'] = np.subtract(df.boll_lb.values, df.close.values)
        df['close_30_sma_close_60_sma'] = np.subtract(df.close_30_sma.values, df.close_60_sma.values)
        df['close_20_sma_close_50_sma'] = np.subtract(df.close_20_sma.values, df.close_50_sma.values)
        df['close_50_sma_close_200_sma'] = np.subtract(df.close_50_sma.values, df.close_200_sma.values)

        daily_changelag_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = df[df.tic == unique_ticker[i]].change.shift(1)
                temp_indicator = pd.DataFrame(temp_indicator)
                daily_changelag_df = daily_changelag_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                print(e)

        fred = Fred(api_key='a2ca2601550a3ac2a1af260112595a8d')
        temp = df['date'].to_frame()
        for series in ['EFFR',
#                        'UNRATE',
#                        'DEXUSEU',
#                        'TEDRATE',
#                        'DTWEXBGS',
#                        'VIXCLS',
#                        'DEXCHUS',
#                        'USRECD',
#                        'DTWEXEMEGS',
#                        'VXEEMCLS',
#                        'A191RL1Q225SBEA',
#                        'GFDEGDQ188S',
#                        'DPCERL1Q225SBEA'
                      ]:

            data = fred.get_series(series)
            data = data.to_frame()
            data = data.reset_index()
            data.columns = [
                "date",
                series
            ]
            data["date"] = data.date.apply(lambda x: x.strftime("%Y-%m-%d"))
            temp_2 = pd.merge(temp, data, how='left', left_on='date', right_on='date')
            temp.fillna(method="ffill")
            df = pd.merge(df, data, how='left', left_on='date', right_on='date')

        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def standardize_dates(self, data):
        date_df = pd.DataFrame({'date_y': pd.bdate_range(start=config.START_DATE,
                                                         end=config.END_DATE,
                                                         freq='B').to_list()})
        holidays = list(mcal.get_calendar('NYSE').holidays().holidays)
        # print(holidays)
        date_df = date_df[~date_df['date_y'].isin(holidays)]
        date_df['date_y'] = pd.to_datetime(date_df['date_y'])
        date_df['date_y'] = date_df.date_y.apply(lambda x: x.strftime("%Y-%m-%d"))
        # print(date_df)

        df = data.copy()
        unique_ticker = df.tic.unique()
        final_df = pd.DataFrame()
        # print(df)
        for i in range(len(unique_ticker)):
            try:
                temp_ticker = df[df.tic == unique_ticker[i]]
                temp_ticker = pd.DataFrame(temp_ticker)
                # print(type(date_df.date_y[0]))
                temp_date_df = pd.merge(date_df, temp_ticker, how='left', left_on='date_y', right_on='date')
                # print(temp_date_df)
                temp_date_df['date'] = temp_date_df['date_y']
                temp_date_df.drop('date_y', axis=1, inplace=True)
                final_df = final_df.append(
                    temp_date_df, ignore_index=True
                )
            except Exception as e:
                print(e)

        final_df = final_df.fillna(0)
        return final_df
