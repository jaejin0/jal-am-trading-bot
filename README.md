# A Trading Bot using Joint-Action Learning with Deep Agent Modeling

In trading, there are market where the value is determined by dividend and expected value in the future. But in Bitcoin, there is no dividend, meaning the dynamics of the price has higher dependency on people's trading strategy than stock market. So, this project is to exploit psychological human's trading strategy as well as traditional Deep Learning-based Trading Bots.

In Multi-agent Reinforcement Learning (MARL), there is an algorithm called Joint Action Learning with Agent Modeling (JAL-AM) which creates a policy network for other agents and perform action to exploit the other.

So in the trading market, where it is basically a Zero Sum game, we can create a policy network which represent the other bots and people's strategies, and have another network which can exploit such strategy.


Action space : Buy 1 Bitcoin, Sell 1 Bitcoin, No-op


## Source
[Multi-Agent Reinforcement Learning Book](https://www.marl-book.com/)
[Bitcoin Historical Dataset on Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd)
