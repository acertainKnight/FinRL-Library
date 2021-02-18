import pandas as pd
import glob
import os
import talib as ta
import numpy as np
from multiprocessing import get_context, cpu_count


def minto15_alpaca(path):
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            path_sub = os.path.join(root, d)  # this is the current subfolder
            for fname in glob.glob(os.path.join(path_sub, '*.csv')):
                print(fname)
                try:
                    df = pd.read_csv(fname)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
                    # print(type(df.timestamp[0]))
                    datetime_index = pd.DatetimeIndex(df.timestamp.values)
                    df = df.set_index(datetime_index)
                    df.drop(['timestamp', 'vwap'], axis=1, inplace=True)
                    # df.set_index('timestamp', inplace=True)
                    df = df.between_time(start_time='09:30', end_time='16:00')
                    temp = pd.DataFrame(index=pd.date_range(start='2009-01-02', end='2021-01-01', freq='min'))
                    temp = temp.between_time(start_time='09:30', end_time='16:00')
                    df = pd.merge(temp, df, how='left', left_index=True, right_index=True)
                    df = df.fillna(method='ffill').fillna(0)
                    resampled_data = df.resample('15T', closed='right', label='right').agg({'open': 'first',
                                                                                              'high': 'max',
                                                                                              'low': 'min',
                                                                                              'close': 'last'}).dropna()
                    resampled_data.to_csv(fname[:-4] + '-15M' + '.csv')
                except:
                    print('Failed')
                # print(resampled_data.head(50))

                # my_list=list(df.columns)
                # print(len(my_list),my_list)


def stack(params):
    path = params[0]
    idx = params[1]
    df_stacked = pd.DataFrame()
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if d != '15min':
                continue
            else:
                path_sub = os.path.join(root, d)  # this is the current subfolder
                ___ = 0
                for fname in glob.glob(os.path.join(path_sub, '*.csv'))[idx[0]:idx[1]]:
                    print(len(glob.glob(os.path.join(path_sub, '*.csv'))[idx[0]:idx[1]]) - ___)
                    df_raw = pd.read_csv(fname)
                    # print(df_raw.columns)
                    df_raw['timestamp'] = pd.to_datetime(df_raw['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S', utc=True)
                    # print(type(df.timestamp[0]))
                    datetime_index = pd.DatetimeIndex(df_raw.timestamp.values)
                    df_raw = df_raw.set_index(datetime_index)
                    df_raw = df_raw.drop('Unnamed: 0', axis=1)
                    df_resample60 = df_raw.resample('60T', closed='right', label='right').agg({'open': 'first',
                                                                                              'high': 'max',
                                                                                              'low': 'min',
                                                                                              'close': 'last'}).dropna()
                    # df_resample24 = df_raw.resample('1D', closed='right', label='right').agg({'open': 'first',
                    #                                                                           'high': 'max',
                    #                                                                           'low': 'min',
                    #                                                                           'close': 'last'}).dropna()

                    df_full = pd.DataFrame(index=df_raw.index.values)
                    for df in [df_raw, df_resample60]:
                        # df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'])
                        _, __, df['macdhist'] = ta.MACD(df['close'])
                        # df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['close'])
                        df['upperband'], _, df['lowerband'] = ta.BBANDS(df['close'])
                        df['price_upperband'] = np.subtract(df.close.values, df.upperband.values)
                        # df['price_middleband'] = np.subtract(df.close.values, df.middleband.values)
                        df['price_lowerband'] = np.subtract(df.close.values, df.lowerband.values)
                        df['ema_9'] = ta.EMA(df['close'], timeperiod=9)
                        df['ema_21'] = ta.EMA(df['close'], timeperiod=21)
                        df['price_ema9'] = np.subtract(df.close.values, df.ema_9.values)
                        # df['price_ema21'] = np.subtract(df.close.values, df.ema_21.values)
                        df['ema9_ema21'] = np.subtract(df.ema_9.values, df.ema_21.values)
                        df['rsi'] = ta.RSI(df['close'])
                        df['rsi_80'] = (df['rsi'] >= 80) * 1
                        df['rsi_20'] = (df['rsi'] <= 20) * 1
                        df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'])
                        df['slowk_slowd'] = np.subtract(df.slowk.values, df.slowd.values)
                        # df['OBV'] = ta.OBV(df['close'], df['volume'])
                        df2 = pd.merge(df, df.diff(), how='left', left_index=True, right_index=True)
                        # df2 = pd.merge(df2, df.diff(27), how='left', left_index=True, right_index=True)
                        df_full = pd.merge(df_full, df2, how='left', left_index=True, right_index=True)
                    df_full = df_full.drop(['timestamp_x', 'timestamp_y'], axis=1)
                    # print(df_full.columns)
                    df_full = df_full.fillna(method='ffill').fillna(0)
                    print('Saving: {}'.format(fname[53:]))
                    # df_full.to_csv(r'/home/nghallmark/FinRL-Library/datasets/ALPACA/{}'.format(fname[53:]))
                    print('Complete')
                    df_stacked = df_stacked.append(df_full)
                    ___ += 1
    return df_stacked





if __name__ == "__main__":
    # minto15_alpaca(r"/Users/Nick/Documents/tic_data/datasets/ALPACA")
    # path = r"/Users/Nick/Documents/tic_data/datasets/ALPACA/15min"
    # path2 = r"/Users/Nick/Documents/tic_data/datasets/ALPACA"
    path = r"/home/nghallmark/FinRL-Library/datasets/ALPACA/15min"
    path2 = r"/home/nghallmark/FinRL-Library/datasets/ALPACA"

    temp = len(glob.glob(os.path.join(path, '*.csv')))
    print(temp)
    print(cpu_count())
    print(int(temp / cpu_count()))
    tempindex_list = list(range(0, temp, int(temp / cpu_count())))
    # print(len(df.tic.unique()))
    # print(tempindex_list)
    index_list = []
    for i in range(len(tempindex_list)):
        # print(i)
        if i == list(range(len(tempindex_list)))[-1]:
            _ = [path2, [tempindex_list[i], 1 + temp]]
        else:
            _ = [path2, [tempindex_list[i], tempindex_list[i + 1]]]
        index_list.append(_)
    with get_context("spawn").Pool() as pool:
        result = pool.map(stack, index_list)
    df_stackFULL = pd.DataFrame()
    for df in result:
        df_stackFULL = df_stackFULL.append(df)
    print('Saving Final')
    df_stackFULL.to_csv(r'/home/nghallmark/FinRL-Library/datasets/alpaca.csv')
