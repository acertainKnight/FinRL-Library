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
    print(config.BRDGWTR_50_TICKER)

    df = YahooDownloader(start_date=config.START_DATE,
                         end_date=config.END_DATE,
                         ticker_list=config.BRDGWTR_50_TICKER).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=True)

    processed = fe.preprocess_data(df)
#     processed = df

    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    information_cols = list(processed)
    information_cols.remove('date')
    information_cols.remove('tic')
    
#     information_cols = ['daily_variance', 'change', 'log_volume', 'close','day', 
#                     'macd', 'rsi_30', 'cci_30', 'dx_30']

    e_train_gym = StockTradingEnvCashpenalty(df = train,initial_amount = 10000,hmax = 100, 
                                    cache_indicator_data=True,
                                    cash_penalty_proportion=0.2, 
                                    daily_information_cols = information_cols, 
                                    print_verbosity = 500, random_start = True)




    e_trade_gym = StockTradingEnvCashpenalty(df = trade,initial_amount = 10000,hmax = 100, 
                                    cash_penalty_proportion=0.2,
                                    cache_indicator_data=True,
                                    daily_information_cols = information_cols, 
                                    print_verbosity = 500, random_start = False)



    n_cores = multiprocessing.cpu_count()
    # n_cores = 24
    print(f"using {n_cores} cores")

    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_multiproc_env(n=n_cores)
    # env_train, _ = e_train_gym.get_sb_env()


    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    # from torch.nn import Softsign, ReLU
    ppo_params ={'n_steps': 256,
                 'ent_coef': 0.0,
                 'learning_rate': 0.0005,
                 'batch_size': 1024,
                'gamma': 0.99}
    
    policy_kwargs = {
    #     "activation_fn": ReLU,
        "net_arch": [1024 for _ in range(10)],
    #     "squash_output": True
    }

    model = agent.get_model("ppo",
                            model_kwargs = ppo_params,
                            policy_kwargs = policy_kwargs, verbose = 1)
    
    model.learn(total_timesteps = 1000000,
                eval_env = env_trade,
                eval_freq = 500,
                log_interval = 1,
                tb_log_name = 'env_cashpenalty_PPO',
                n_eval_episodes = 1)
    
    model.save("different1_PPO.model")

    # model = model.load("scaling_reward.model", env = env_train)
#     for strat in ["a2c", "ddpg", "td3", "sac", "ppo"]:
#         model = agent.get_model(strat, verbose=0)

#         model.learn(total_timesteps = 5000000,
#                     eval_env = env_trade,
#                     eval_freq = 500,
#                     log_interval = 1,
#                     tb_log_name = 'env_cashpenalty_{}'.format(strat),
#                     n_eval_episodes = 1)
#         model.save("different1_{}.model".format(strat))

    e_trade_gym.hmax = 5000
    
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,
                                                           environment=e_trade_gym)
    
    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_account_value,
                                    value_col_name='total_assets')
    
    print("==============Compare to DJIA===========")
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker='^DJI',
                  baseline_start=config.START_TRADE_DATE,
                  baseline_end=config.END_DATE, value_col_name='total_assets')

if __name__ == "__main__":
    main()