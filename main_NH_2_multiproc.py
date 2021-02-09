import sys

sys.path.append("..")
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors_multiproc import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.model.models_multiproc import DRLAgent, DRLEnsembleAgent
from finrl.trade.backtest import backtest_plot, backtest_stats
import os
import multiprocessing


def main():
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    print(config.START_DATE)
    print(config.END_DATE)
    print(config.BRDGWTR_50_TICKER)

    df = YahooDownloader(start_date=config.START_DATE,
                         end_date=config.END_DATE,
                         ticker_list=config.PENNY_STOCKS).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)
    information_cols = list(processed)
    information_cols.remove('date')
    information_cols.remove('tic')

    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(information_cols) * stock_dimension
    print("Stock Dimension: {}, State Space: {}".format(stock_dimension, state_space))

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 5000,
        # Since in Indonesia the minimum number of shares per trx is 100, then we scaled the initial amount by dividing it with 100
        "buy_cost_pct": 0.00,  # IPOT has 0.19% buy cost
        "sell_cost_pct": 0.00,  # IPOT has 0.29% sell cost
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": information_cols,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 5

    }

    rebalance_window = 21  # rebalance_window is the number of days to retrain the model
    validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
    train_start = config.START_DATE
    train_end = config.START_TRADE_DATE
    val_test_start = config.START_TRADE_DATE
    val_test_end = config.END_DATE

    ensemble_agent = DRLEnsembleAgent(df=processed,
                                      train_period=(train_start, train_end),
                                      val_test_period=(val_test_start, val_test_end),
                                      rebalance_window=rebalance_window,
                                      validation_window=validation_window,
                                      **env_kwargs)

    A2C_model_kwargs = {
        'n_steps': 5,
        'ent_coef': 0.01,
        'learning_rate': 0.0005
    }

    PPO_model_kwargs = {
        "ent_coef": 0.01,
        "n_steps": 2048,
        "learning_rate": 0.00025,
        "batch_size": 128
    }

    DDPG_model_kwargs = {
        "action_noise": "ornstein_uhlenbeck",
        "buffer_size": 50000,
        "learning_rate": 0.000005,
        "batch_size": 128
    }

    TD3_model_kwargs = {
        "batch_size": 100,
        "buffer_size": 1000000,
        "learning_rate": 0.001
    }

    timesteps_dict = {'a2c': 100000,
                      'ppo': 100000,
                      'ddpg': 50000,
                      'td3': 50000
                      }

    df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                      PPO_model_kwargs,
                                                      DDPG_model_kwargs,
                                                      TD3_model_kwargs,
                                                      timesteps_dict)

    print(df_summary)

    unique_trade_date = processed[
        (processed.date > val_test_start) & (processed.date <= val_test_end)].date.unique()

    df_trade_date = pd.DataFrame({'datadate': unique_trade_date})

    df_account_value = pd.DataFrame()
    for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
        print(rebalance_window + validation_window)
        print(len(unique_trade_date) + 1)
        print(rebalance_window)
        try:
            temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble', i))
            df_account_value = df_account_value.append(temp, ignore_index=True)
        except:
            break
    sharpe = (252 ** 0.5) * df_account_value.account_value.pct_change(
        1).mean() / df_account_value.account_value.pct_change(1).std()
    print('Sharpe Ratio: ', sharpe)
    df_account_value = df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

    df_account_value.account_value.plot()

    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    print("==============Compare to IHSG===========")
    backtest_plot(df_account_value,
                  baseline_ticker='^DJI',
                  baseline_start=df_account_value.loc[0, 'date'],
                  baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])


if __name__ == "__main__":
    main()
