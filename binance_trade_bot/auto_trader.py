from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
from sqlalchemy.orm import Session

from .binance_api_manager import BinanceAPIManager
from .config import Config
from .database import Database, LogScout
from .logger import Logger
from .models import Coin, CoinValue, Pair


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


class AutoTrader:
    def __init__(self, binance_manager: BinanceAPIManager, database: Database, logger: Logger, config: Config):
        self.manager = binance_manager
        self.db = database
        self.logger = logger
        self.config = config
        self.failed_buy_order = False

    def initialize(self):
        self.initialize_trade_thresholds()

    def transaction_through_bridge(self, pair: Pair, sell_price: float, buy_price: float):
        """
        Jump from the source coin to the destination coin through bridge coin
        """
        self.manager.cancel_previous_orders(pair.from_coin, self.config.BRIDGE_SYMBOL)

        can_sell = False
        balance = self.manager.get_currency_balance(pair.from_coin.symbol)

        if balance and balance * sell_price > self.manager.get_min_notional(
            pair.from_coin.symbol, self.config.BRIDGE.symbol
        ):
            can_sell = True
        else:
            self.logger.info("Skipping sell")

        if can_sell and self.manager.sell_alt(pair.from_coin, self.config.BRIDGE, sell_price) is None:
            self.logger.info("Couldn't sell, going back to scouting mode...")
            return None

        result = self.manager.buy_alt(pair.to_coin, self.config.BRIDGE, buy_price)
        if result is not None:
            self.db.set_current_coin(pair.to_coin)
            price = result.price
            if abs(price) < 1e-15:
                price = result.cumulative_quote_qty / result.cumulative_filled_quantity

            self.update_trade_threshold(pair.to_coin, price)
            self.failed_buy_order = False
            return result

        self.logger.info("Couldn't buy, going back to scouting mode...")
        self.failed_buy_order = True
        return None

    def update_trade_threshold(self, coin: Coin, coin_price: float):
        """
        Update all the coins with the threshold of buying the current held coin
        """

        if coin_price is None or coin_price == 0.0:
            self.logger.info("Skipping update... current coin {} not found".format(coin + self.config.BRIDGE))
            return

        session: Session
        with self.db.db_session() as session:
            for pair in session.query(Pair).filter(Pair.to_coin == coin):
                from_coin_price = self.manager.get_sell_price(pair.from_coin + self.config.BRIDGE)

                if from_coin_price is None:
                    self.logger.info(
                        "Skipping update for coin {} not found".format(pair.from_coin + self.config.BRIDGE)
                    )
                    continue

                # check if we hold above min_notional coins of from_coin. If so skip ratio update.
                from_coin_balance = self.manager.get_currency_balance(pair.from_coin.symbol)
                min_notional = self.manager.get_min_notional(pair.from_coin.symbol, self.config.BRIDGE.symbol)
                if from_coin_price * from_coin_balance > min_notional:
                    continue

                pair.ratio = from_coin_price / coin_price

    def initialize_trade_thresholds(self):
        """
        Initialize the buying threshold of all the coins for trading between them
        """
        session: Session
        with self.db.db_session() as session:
            pairs = session.query(Pair).filter(Pair.ratio.is_(None)).all()
            grouped_pairs = defaultdict(list)
            for pair in pairs:
                if pair.from_coin.enabled and pair.to_coin.enabled:
                    grouped_pairs[pair.from_coin.symbol].append(pair)
            for from_coin_symbol, group in grouped_pairs.items():
                self.logger.info(f"Initializing {from_coin_symbol} vs [{', '.join([p.to_coin.symbol for p in group])}]")
                for pair in group:
                    from_coin_price = self.manager.get_sell_price(pair.from_coin + self.config.BRIDGE)
                    if from_coin_price is None:
                        self.logger.info(
                            "Skipping initializing {}, symbol not found".format(pair.from_coin + self.config.BRIDGE)
                        )
                        continue

                    to_coin_price = self.manager.get_buy_price(pair.to_coin + self.config.BRIDGE)
                    if to_coin_price is None or to_coin_price == 0.0:
                        self.logger.info(
                            "Skipping initializing {}, symbol not found".format(pair.to_coin + self.config.BRIDGE)
                        )
                        continue

                    pair.ratio = from_coin_price / to_coin_price

    def scout(self):
        """
        Scout for potential jumps from the current coin to another coin
        """
        raise NotImplementedError()

    # pylint: disable=dangerous-default-value
    def _get_ratios(self, coin: Coin, coin_price, excluded_coins: List[Coin] = []):
        """
        Given a coin, get the current price ratio for every other enabled coin
        """
        ratio_dict: Dict[Pair, float] = {}
        prices: Dict[str, float] = {}

        scout_logs = []
        excluded_coin_symbols = [c.symbol for c in excluded_coins]
        for pair in self.db.get_pairs_from(coin):
            # skip excluded coins
            if pair.to_coin.symbol in excluded_coin_symbols:
                continue

            optional_coin_price = self.manager.get_buy_price(pair.to_coin + self.config.BRIDGE)
            prices[pair.to_coin_id] = optional_coin_price

            if optional_coin_price is None or optional_coin_price == 0.0:
                self.logger.info(
                    "Skipping scouting... optional coin {} not found".format(pair.to_coin + self.config.BRIDGE)
                )
                continue

            scout_logs.append(LogScout(pair, pair.ratio, coin_price, optional_coin_price))

            # Obtain (current coin)/(optional coin)
            coin_opt_coin_ratio = coin_price / optional_coin_price

            # Fees
            from_fee = self.manager.get_fee(pair.from_coin, self.config.BRIDGE, True)
            to_fee = self.manager.get_fee(pair.to_coin, self.config.BRIDGE, False)
            transaction_fee = from_fee + to_fee - from_fee * to_fee

            if self.config.USE_MARGIN == "yes":
                ratio_dict[pair] = (
                    (1 - transaction_fee) * coin_opt_coin_ratio / pair.ratio - 1 - self.config.SCOUT_MARGIN / 100
                )
            else:
                ratio_dict[pair] = (
                    coin_opt_coin_ratio - transaction_fee * self.config.SCOUT_MULTIPLIER * coin_opt_coin_ratio
                ) - pair.ratio
        self.db.batch_log_scout(scout_logs)
        return (ratio_dict, prices)

    # pylint: disable=dangerous-default-value
    def _jump_to_best_coin(self, coin: Coin, coin_price: float, excluded_coins: List[Coin] = []):
        """
        Given a coin, search for a coin to jump to
        """
        ratio_dict, prices = self._get_ratios(coin, coin_price, excluded_coins)

        # keep only ratios bigger than zero
        ratio_dict = {k: v for k, v in ratio_dict.items() if v > 0}

        # if we have any viable options, pick the one with the biggest ratio
        if ratio_dict:
            best_pair = max(ratio_dict, key=ratio_dict.get)
            self.logger.info(f"Will be jumping from {coin} to {best_pair.to_coin_id}")
            self.transaction_through_bridge(best_pair, coin_price, prices[best_pair.to_coin_id])

    def bridge_scout(self):
        """
        If we have any bridge coin leftover, buy a coin with it that we won't immediately trade out of
        """
        bridge_balance = self.manager.get_currency_balance(self.config.BRIDGE.symbol)

        for coin in self.db.get_coins():
            current_coin_price = self.manager.get_sell_price(coin + self.config.BRIDGE)

            if current_coin_price is None:
                continue

            ratio_dict, _ = self._get_ratios(coin, current_coin_price)
            if not any(v > 0 for v in ratio_dict.values()):
                # There will only be one coin where all the ratios are negative. When we find it, buy it if we can
                if bridge_balance > self.manager.get_min_notional(coin.symbol, self.config.BRIDGE.symbol):
                    self.logger.info(f"Will be purchasing {coin} using bridge coin")
                    result = self.manager.buy_alt(
                        coin, self.config.BRIDGE, self.manager.get_sell_price(coin + self.config.BRIDGE)
                    )
                    if result is not None:
                        self.db.set_current_coin(coin)
                        self.failed_buy_order = False
                        return coin

                    self.failed_buy_order = True
        return None

    def update_values(self):
        """
        Log current value state of all altcoin balances against BTC and USDT in DB.
        """
        now = datetime.now()

        coins = self.db.get_coins(True)
        cv_batch = []
        for coin in coins:
            balance = self.manager.get_currency_balance(coin.symbol)

            current_coin = self.db.get_current_coin()

            if coin.symbol == current_coin.symbol:
                orders = self.manager.get_pair_orders(current_coin.symbol, self.config.BRIDGE_SYMBOL)
                for order in orders:
                    balance += float(order["origQty"])

            if balance == 0.0:
                continue
            usd_value = self.manager.get_ticker_price(coin + self.config.BRIDGE_SYMBOL)
            btc_value = self.manager.get_ticker_price(coin + "BTC")
            cv = CoinValue(coin, balance, usd_value, btc_value, datetime=now)
            cv_batch.append(cv)
        self.db.batch_update_coin_values(cv_batch)

    def update_current_value(self):
        """
        Log current value state of all altcoin balances against BTC and USDT in DB.
        """
        now = datetime.now()

        coin = current_coin = self.db.get_current_coin()
        cv_batch = []
        balance = self.manager.get_currency_balance(coin.symbol)

        current_coin = self.db.get_current_coin()

        if coin.symbol == current_coin.symbol:
            orders = self.manager.get_pair_orders(current_coin.symbol, self.config.BRIDGE_SYMBOL)
            for order in orders:
                balance += float(order["origQty"])

        if balance == 0.0:
            return

        usd_value = self.manager.get_ticker_price(coin + self.config.BRIDGE_SYMBOL)
        btc_value = self.manager.get_ticker_price(coin + "BTC")
        cv = CoinValue(coin, balance, usd_value, btc_value, datetime=now)
        cv_batch.append(cv)
        self.db.batch_update_coin_values(cv_batch)

    def update_orders(self):
        """
        Update trading strategy while you are in a coin
        """

        current_coin = self.db.get_current_coin()
        balance = self.manager.get_currency_balance(current_coin.symbol)

        orders = self.manager.get_pair_orders(current_coin.symbol, self.config.BRIDGE_SYMBOL)
        for order in orders:
            order_qty = float(order["origQty"])
            balance += order_qty

        self.logger.debug(f"Updating existing orders {current_coin} with balance {balance}")
        if balance == 0:
            return
        usd_value = self.manager.get_ticker_price(current_coin + "USDT")

        if current_coin.symbol != self.config.BRIDGE_SYMBOL:
            self.logger.debug(f"Resolving orders {orders} ")
            if orders is None:
                return

            if isinstance(orders, list) and len(orders) > 0:
                for order in orders:
                    if order["side"] == self.manager.binance_client.SIDE_SELL and float(order["stopPrice"]) > usd_value:
                        self.manager.cancel_order(order["symbol"], order["orderId"])
                        continue

                    if order["side"] == self.manager.binance_client.SIDE_BUY and float(order["stopPrice"]) < usd_value:
                        self.manager.cancel_order(order["symbol"], order["orderId"])
                        continue

                    margin_coin, update_buy_mul, update_sell_mul = self.db.get_current_margins()
                    if margin_coin != current_coin or update_buy_mul is None or update_sell_mul is None:
                        update_buy_mul = self.config.UPDATE_BUY_MUL
                        update_sell_mul = self.config.UPDATE_SELL_MUL

                    if (
                        "stopPrice" in order
                        and order["side"] == self.manager.binance_client.SIDE_SELL
                        and float(order["stopPrice"]) > 0.0
                        and float(order["stopPrice"]) < usd_value * (1 - update_sell_mul / 100)
                    ):
                        self.logger.debug("Updating stop loss sell to " + str(usd_value * (1 - update_sell_mul / 100)))
                        self.manager.cancel_order(order["symbol"], order["orderId"])
                        self.manager.set_sell_stop_loss_order(
                            current_coin.symbol,
                            self.config.BRIDGE_SYMBOL,
                            usd_value,
                            float(order["origQty"]),
                            mul=self.config.UPDATE_SELL_MUL,
                        )
                    elif (
                        "stopPrice" in order
                        and order["side"] == self.manager.binance_client.SIDE_BUY
                        and float(order["stopPrice"]) > 0.0
                        and float(order["stopPrice"]) > usd_value * (1 + update_buy_mul / 100)
                    ):
                        self.logger.debug("Updating stop loss buy to " + str(usd_value * (1 + update_buy_mul / 100)))
                        self.manager.cancel_order(order["symbol"], order["orderId"])
                        self.manager.set_buy_stop_loss_order(
                            current_coin.symbol,
                            self.config.BRIDGE_SYMBOL,
                            usd_value,
                            mul=self.config.UPDATE_BUY_MUL,
                        )
            else:
                coin_balance = 0
                bridge_balance = 0

                try:
                    coin_balance = float(self.manager.binance_client.get_asset_balance(current_coin.symbol)["free"])
                    bridge_balance = float(
                        self.manager.binance_client.get_asset_balance(self.config.BRIDGE_SYMBOL)["free"]
                    )
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.warning(f"Unexpected Error: {e}")

                if coin_balance * usd_value > 1:
                    self.logger.info(
                        "Set stop loss order to sell at " + str(usd_value * (1 - self.config.FIRST_SELL_MUL / 100))
                    )
                    self.manager.set_sell_stop_loss_order(
                        current_coin.symbol, self.config.BRIDGE_SYMBOL, usd_value, mul=self.config.FIRST_SELL_MUL
                    )

                elif bridge_balance > 1:
                    self.logger.info(
                        "Set stop loss order to buy at " + str(usd_value * (1 + self.config.FIRST_BUY_MUL / 100))
                    )
                    self.manager.set_buy_stop_loss_order(
                        current_coin.symbol, self.config.BRIDGE_SYMBOL, usd_value, mul=self.config.FIRST_BUY_MUL
                    )

    def update_trend_margins(self):
        """
        Log current value state of all altcoin balances against BTC and USDT in DB.
        """

        current_coin = self.db.get_current_coin()
        margin_coin, origbuy, origsell = self.db.get_current_margins()
        history = self.db.get_current_history(margin_coin)
        usd_value = self.manager.get_ticker_price(current_coin + self.config.BRIDGE_SYMBOL)

        if margin_coin is None:
            self.db.set_current_margins(
                current_coin, usd_value, self.config.UPDATE_BUY_MUL, self.config.UPDATE_SELL_MUL
            )
            return
        if current_coin.symbol != margin_coin:
            self.db.set_current_margins(current_coin, usd_value, origbuy, origsell)
            return

        # Change values based on trend
        movingStep = 0.1
        increasing = np.all(np.diff(moving_average(np.array(history), n=50)) > 0)

        # if self.config.UPDATE_BUY_MUL != origbuy or self.config.UPDATE_SELL_MUL != origsell:
        #    buy = self.config.UPDATE_BUY_MUL
        #    sell = self.config.UPDATE_SELL_MUL
        if not increasing:
            buy = origbuy + movingStep
            if buy > 4.5:
                buy = 4.5
            sell = origsell - movingStep
            if sell < 0.5:
                sell = 0.5
        else:
            buy = origbuy - movingStep
            if buy < 0.5:
                buy = 0.5
            sell = origsell + movingStep
            if sell > 4.5:
                sell = 4.5

        if origbuy != buy or origsell != sell:
            self.logger.info(
                "Set current margins for "
                + current_coin.symbol
                + "\n $"
                + str(usd_value)
                + (" goes up." if increasing else " goes down.")
                + " Buy: "
                + str(buy)
                + "%, Sell: "
                + str(sell)
                + "%\n last10vals: "
                + str(history)
            )

        self.db.set_current_margins(current_coin, usd_value, buy, sell)
