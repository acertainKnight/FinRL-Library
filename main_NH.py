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
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.model.models import DRLAgent, DRLEnsembleAgent
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
    print(config.DOW_30_TICKER)

    df = YahooDownloader(start_date=config.START_DATE,
                         end_date=config.END_DATE,
                         ticker_list=config.DOW_30_TICKER).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)

    processed['log_volume'] = np.log(processed.volume * processed.close)
    processed['change'] = np.divide(np.subtract(processed.close.values, processed.open.values),
                                    processed.close.values)
    processed['daily_variance'] = np.divide(np.subtract(processed.high.values, processed.low.values),
                                            processed.close.values)

    train = data_split(processed, '2009-01-01', '2016-01-01')
    trade = data_split(processed, '2016-01-01', '2021-01-01')

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'day',
                        'macd', 'rsi_30', 'cci_30', 'dx_30']

    e_train_gym = StockTradingEnvCashpenalty(df=train, initial_amount=1e5, hmax=5000,
                                             cache_indicator_data=True,
                                             cash_penalty_proportion=0.1,
                                             daily_information_cols=information_cols,
                                             print_verbosity=1000, random_start=True)

    e_trade_gym = StockTradingEnvCashpenalty(df=trade, initial_amount=1e5, hmax=5000,
                                             cash_penalty_proportion=0.1,
                                             cache_indicator_data=True,
                                             daily_information_cols=information_cols,
                                             print_verbosity=1000, random_start=False)



    n_cores = multiprocessing.cpu_count()
    # n_cores = 24
    print(f"using {n_cores} cores")

    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_multiproc_env(n=n_cores)
    # env_train, _ = e_train_gym.get_sb_env()


    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    # ppo_params = {'n_steps': 256,
    #               'ent_coef': 0.0,
    #               'learning_rate': 0.000005,
    #               'batch_size': 1024,
    #               'gamma': 0.99}
    #
    # policy_kwargs = {
    #     #     "activation_fn": ReLU,
    #     "net_arch": [1024 for _ in range(10)],
    #     #     "squash_output": True
    # }
    for strat in ["a2c", "ddpg", "td3", "sac", "ppo"]:
        print('Training: {}'.format(strat))
        model = agent.get_model(strat, verbose=0)

        model.learn(total_timesteps=1000000,
                    eval_env=env_trade,
                    log_interval=1,
                    tb_log_name='env_cashpenalty_{}'.format(strat))

        model.save("{}.model".format(strat))

    # trade.head()
    #
    # e_trade_gym.hmax = 5000
    #
    # print(len(e_trade_gym.dates))
    #
    # df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
    #
    # df_actions.head()
    #
    # df_account_value.shape
    #
    # df_account_value.head(50)
    #
    # print("==============Get Backtest Results===========")
    # perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name='total_assets')
    #
    # print("==============Compare to DJIA===========")
    # # S&P 500: ^GSPC
    # # Dow Jones Index: ^DJI
    # # NASDAQ 100: ^NDX
    # backtest_plot(df_account_value,
    #               baseline_ticker='^DJI',
    #               baseline_start='2016-01-01',
    #               baseline_end='2021-01-01', value_col_name='total_assets')

if __name__ == "__main__":
    main()