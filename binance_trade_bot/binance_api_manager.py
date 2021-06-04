import decimal
import math
import time
import traceback
from datetime import datetime
from typing import Dict, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from cachetools import TTLCache, cached

from .binance_stream_manager import BinanceCache, BinanceOrder, BinanceStreamManager, OrderGuard
from .config import Config
from .database import Database
from .logger import Logger
from .models import Coin


def price_decimals(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    and returns number of decimals
    """

    # create a new context for this task
    ctx = decimal.Context()

    # 20 digits should be enough for everyone :D
    ctx.prec = 20

    d1 = ctx.create_decimal(repr(f))
    floatstring = format(d1, "f")

    decimals = 0

    if "." in floatstring:
        decimals = len(floatstring.split(".")[1])

    return decimals


def now():
    return datetime.now()


class BinanceAPIManager:
    def __init__(self, config: Config, db: Database, logger: Logger):
        # initializing the client class calls `ping` API endpoint, verifying the connection
        self.binance_client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET_KEY,
            tld=config.BINANCE_TLD,
        )
        self.db = db
        self.logger = logger
        self.config = config

        self.cache = BinanceCache()
        self.stream_manager: Optional[BinanceStreamManager] = None
        self.setup_websockets()

    def setup_websockets(self):
        self.stream_manager = BinanceStreamManager(
            self.cache,
            self.config,
            self.binance_client,
            self.logger,
        )

    @cached(cache=TTLCache(maxsize=1, ttl=43200))
    def get_trade_fees(self) -> Dict[str, float]:
        return {ticker["symbol"]: float(ticker["takerCommission"]) for ticker in self.binance_client.get_trade_fee()}

    @cached(cache=TTLCache(maxsize=1, ttl=60))
    def get_using_bnb_for_fees(self):
        return self.binance_client.get_bnb_burn_spot_margin()["spotBNBBurn"]

    def get_fee(self, origin_coin: Coin, target_coin: Coin, selling: bool):
        if self.config.TRADE_FEE != "auto":
            return float(self.config.TRADE_FEE)

        base_fee = self.get_trade_fees()[origin_coin + target_coin]
        if not self.get_using_bnb_for_fees():
            return base_fee

        # The discount is only applied if we have enough BNB to cover the fee
        amount_trading = (
            self._sell_quantity(origin_coin.symbol, target_coin.symbol)
            if selling
            else self._buy_quantity(origin_coin.symbol, target_coin.symbol)
        )

        fee_amount = amount_trading * base_fee * 0.75
        if origin_coin.symbol == "BNB":
            fee_amount_bnb = fee_amount
        else:
            origin_price = self.get_ticker_price(origin_coin + Coin("BNB"))
            if origin_price is None:
                return base_fee
            fee_amount_bnb = fee_amount * origin_price

        bnb_balance = self.get_currency_balance("BNB")

        if bnb_balance >= fee_amount_bnb:
            return base_fee * 0.75
        return base_fee

    def get_account(self):
        """
        Get account information
        """
        return self.binance_client.get_account()

    def get_buy_price(self, ticker_symbol: str):
        price_type = self.config.PRICE_TYPE
        if price_type == Config.PRICE_TYPE_ORDERBOOK:
            return self.get_ask_price(ticker_symbol)

        return self.get_ticker_price(ticker_symbol)

    def get_sell_price(self, ticker_symbol: str):
        price_type = self.config.PRICE_TYPE
        if price_type == Config.PRICE_TYPE_ORDERBOOK:
            return self.get_bid_price(ticker_symbol)

        return self.get_ticker_price(ticker_symbol)

    def get_ticker_price(self, ticker_symbol: str):
        """
        Get ticker price of a specific coin
        """
        price = self.cache.ticker_values.get(ticker_symbol, None)
        if price is None and ticker_symbol not in self.cache.non_existent_tickers:
            self.cache.ticker_values = {
                ticker["symbol"]: float(ticker["price"]) for ticker in self.binance_client.get_symbol_ticker()
            }
            self.logger.debug(f"Fetched all ticker prices: {self.cache.ticker_values}")
            price = self.cache.ticker_values.get(ticker_symbol, None)
            if price is None:
                self.logger.info(f"Ticker does not exist: {ticker_symbol} - will not be fetched from now on")
                self.cache.non_existent_tickers.add(ticker_symbol)

        return price

    def get_ask_price(self, ticker_symbol: str):
        """
        Get best ask price of a specific coin
        """
        price = self.cache.ticker_values_ask.get(ticker_symbol, None)
        if price is None and ticker_symbol not in self.cache.non_existent_tickers:
            try:
                ticker = self.binance_client.get_orderbook_ticker(symbol=ticker_symbol)
                price = float(ticker["askPrice"])
            except BinanceAPIException as e:
                if e.code == -1121:  # invalid symbol
                    price = None
                else:
                    raise e
            if price is None:
                self.logger.info(f"Ticker does not exist: {ticker_symbol} - will not be fetched from now on")
                self.cache.non_existent_tickers.add(ticker_symbol)

        return price

    def get_bid_price(self, ticker_symbol: str):
        """
        Get best bid price of a specific coin
        """
        price = self.cache.ticker_values_bid.get(ticker_symbol, None)
        if price is None and ticker_symbol not in self.cache.non_existent_tickers:
            try:
                ticker = self.binance_client.get_orderbook_ticker(symbol=ticker_symbol)
                price = float(ticker["bidPrice"])
            except BinanceAPIException as e:
                if e.code == -1121:  # invalid symbol
                    price = None
                else:
                    raise e
            if price is None:
                self.logger.info(f"Ticker does not exist: {ticker_symbol} - will not be fetched from now on")
                self.cache.non_existent_tickers.add(ticker_symbol)

        return price

    def get_currency_balance(self, currency_symbol: str, force=False) -> float:
        """
        Get balance of a specific coin
        """
        with self.cache.open_balances() as cache_balances:
            balance = cache_balances.get(currency_symbol, None)
            if force or balance is None:
                cache_balances.clear()
                cache_balances.update(
                    {
                        currency_balance["asset"]: float(currency_balance["free"])
                        for currency_balance in self.binance_client.get_account()["balances"]
                    }
                )
                self.logger.debug(f"Fetched all balances: {cache_balances}")
                if currency_symbol not in cache_balances:
                    cache_balances[currency_symbol] = 0.0
                    return 0.0
                return cache_balances.get(currency_symbol, 0.0)

            return balance

    def retry(self, func, *args, **kwargs):
        time.sleep(1)
        attempts = 0
        while attempts < 20:
            try:
                return func(*args, **kwargs)
            except Exception:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to Buy/Sell. Trying Again (attempt {attempts}/20)")
                if attempts == 0:
                    self.logger.warning(traceback.format_exc())
                attempts += 1
        return None

    def get_symbol_filter(self, origin_symbol: str, target_symbol: str, filter_type: str):
        return next(
            _filter
            for _filter in self.binance_client.get_symbol_info(origin_symbol + target_symbol)["filters"]
            if _filter["filterType"] == filter_type
        )

    @cached(cache=TTLCache(maxsize=2000, ttl=43200))
    def get_alt_tick(self, origin_symbol: str, target_symbol: str):
        step_size = self.get_symbol_filter(origin_symbol, target_symbol, "LOT_SIZE")["stepSize"]
        if step_size.find("1") == 0:
            return 1 - step_size.find(".")
        return step_size.find("1") - 1

    @cached(cache=TTLCache(maxsize=2000, ttl=43200))
    def get_min_notional(self, origin_symbol: str, target_symbol: str):
        return float(self.get_symbol_filter(origin_symbol, target_symbol, "MIN_NOTIONAL")["minNotional"])

    def set_sell_stop_loss_order(
        self, origin_symbol: str, target_symbol: str, price: float, order_quantity: float = 0.0, mul: float = None
    ):
        """
        Set a sell stop less order
        """
        precision = price_decimals(price)
        order = None
        while order is None:
            try:
                order_quantity = (
                    self._sell_quantity(origin_symbol, target_symbol)
                    if float(order_quantity) <= 0.0
                    else order_quantity
                )

                multiplier = mul if mul else 0.0

                order = self.binance_client.create_order(
                    symbol=origin_symbol + target_symbol,
                    quantity=order_quantity,
                    type=self.binance_client.ORDER_TYPE_STOP_LOSS_LIMIT,
                    price=round(price * (1 - multiplier / 100), precision),
                    stopPrice=round(price * (1 - multiplier / 100), precision),
                    side=self.binance_client.SIDE_SELL,
                    timeInForce="GTC",
                )

            except BinanceAPIException as e:
                self.logger.info(e)
                time.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"Unexpected Error: {e}")

    def set_buy_stop_loss_order(
        self, origin_symbol: str, target_symbol: str, from_coin_price: float = None, mul: float = None
    ):
        """
        Set a buy stop less order
        """

        with self.cache.open_balances() as balances:
            balances.clear()

        target_balance = self.get_currency_balance(target_symbol)
        from_coin_price = from_coin_price or self.get_ticker_price(origin_symbol + target_symbol)

        precision = price_decimals(from_coin_price)
        multiplier = mul if mul else 0.0
        price = round(from_coin_price * (1 + multiplier / 100), precision)
        order_quantity = self._buy_quantity(origin_symbol, target_symbol, target_balance, price)

        order = None
        while order is None:
            try:
                order = self.binance_client.create_order(
                    symbol=origin_symbol + target_symbol,
                    quantity=order_quantity,
                    type=self.binance_client.ORDER_TYPE_STOP_LOSS_LIMIT,
                    price=price,
                    stopPrice=price,
                    side=self.binance_client.SIDE_BUY,
                    timeInForce="GTC",
                )

            except BinanceAPIException as e:
                self.logger.info(e)
                time.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"Unexpected Error: {e}")

    def _wait_for_order(
        self, order_id, origin_symbol: str, target_symbol: str, order_quantity: str
    ) -> Optional[BinanceOrder]:  # pylint: disable=unsubscriptable-object
        while True:
            order_status: BinanceOrder = self.cache.orders.get(order_id, None)
            if order_status is not None:
                break
            self.logger.debug(f"Waiting for order {order_id} to be created")
            time.sleep(1)

        self.logger.debug(f"Order created: {order_status}")

        while order_status.status != "FILLED":
            try:
                order_status = self.cache.orders.get(order_id, None)

                self.logger.debug(f"Waiting for order {order_id} to be filled")

                if self._should_cancel_order(order_status):
                    cancel_order = None
                    while cancel_order is None:
                        cancel_order = self.binance_client.cancel_order(
                            symbol=origin_symbol + target_symbol, orderId=order_id
                        )
                    self.logger.info("Order timeout, canceled...")

                    # sell partially
                    if order_status.status == "PARTIALLY_FILLED" and order_status.side == "BUY":
                        self.logger.info("Sell partially filled amount")

                        order_quantity = self._sell_quantity(origin_symbol, target_symbol)
                        partially_order = None
                        while partially_order is None:
                            partially_order = self.binance_client.order_market_sell(
                                symbol=origin_symbol + target_symbol, quantity=order_quantity
                            )

                    self.logger.info("Going back to scouting mode...")
                    return None

                if order_status.status == "CANCELED":
                    self.logger.info("Order is canceled, going back to scouting mode...")
                    return None

                time.sleep(1)
            except BinanceAPIException as e:
                self.logger.info(e)
                time.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.info(f"Unexpected Error: {e}")
                time.sleep(1)

        self.logger.debug(f"Order filled: {order_status}")
        return order_status

    def wait_for_order(
        self, order_id, origin_symbol: str, target_symbol: str, order_guard: OrderGuard, order_quantity: str
    ) -> Optional[BinanceOrder]:  # pylint: disable=unsubscriptable-object
        with order_guard:
            return self._wait_for_order(order_id, origin_symbol, target_symbol, order_quantity)

    def _should_cancel_order(self, order_status):
        minutes = (time.time() - order_status.time / 1000) / 60
        timeout = 0

        if order_status.side == "SELL":
            timeout = float(self.config.SELL_TIMEOUT)
        else:
            timeout = float(self.config.BUY_TIMEOUT)

        if timeout and minutes > timeout and order_status.status == "NEW":
            return True

        if timeout and minutes > timeout and order_status.status == "PARTIALLY_FILLED":
            if order_status.side == "SELL":
                return True

            if order_status.side == "BUY":
                current_price = self.get_buy_price(order_status.symbol)
                if float(current_price) * (1 - 0.001) > float(order_status.price):
                    return True

        return False

    def buy_alt(self, origin_coin: Coin, target_coin: Coin, buy_price: float) -> BinanceOrder:
        return self.retry(self._buy_alt, origin_coin, target_coin, buy_price)

    def _buy_quantity(
        self, origin_symbol: str, target_symbol: str, target_balance: float = None, from_coin_price: float = None
    ):
        target_balance = target_balance or self.get_currency_balance(target_symbol)
        from_coin_price = from_coin_price or self.get_buy_price(origin_symbol + target_symbol)

        origin_tick = self.get_alt_tick(origin_symbol, target_symbol)
        return math.floor(target_balance * 10 ** origin_tick / from_coin_price) / float(10 ** origin_tick)

    @staticmethod
    def float_as_decimal_str(num: float):
        return f"{num:0.08f}".rstrip("0").rstrip(".")  # remove trailing zeroes too

    def _make_order(
        self,
        side: str,
        coinSymbol: str,
        quantity: float,
        price: float,
        quote_quantity: float,
    ):
        params = {
            "symbol": coinSymbol,
            "side": side,
            "quantity": self.float_as_decimal_str(quantity),
            "type": self.config.BUY_ORDER_TYPE if side == Client.SIDE_BUY else self.config.SELL_ORDER_TYPE,
        }
        if params["type"] == Client.ORDER_TYPE_LIMIT:
            params["timeInForce"] = self.binance_client.TIME_IN_FORCE_GTC
            params["price"] = self.float_as_decimal_str(price)
        elif side == Client.SIDE_BUY:
            del params["quantity"]
            params["quoteOrderQty"] = self.float_as_decimal_str(quote_quantity)
        return self.binance_client.create_order(**params)

    def _buy_alt(self, origin_coin: Coin, target_coin: Coin, buy_price: float):  # pylint: disable=too-many-locals
        """
        Buy altcoin
        """
        origin_symbol = origin_coin.symbol
        target_symbol = target_coin.symbol

        with self.cache.open_balances() as balances:
            balances.clear()

        origin_balance = self.get_currency_balance(origin_symbol)
        target_balance = self.get_currency_balance(target_symbol)
        from_coin_price = self.get_buy_price(origin_symbol + target_symbol)

        buy_max_price_change = float(self.config.BUY_MAX_PRICE_CHANGE)
        if from_coin_price > buy_price * (1.0 + buy_max_price_change):
            self.logger.info("Buy price became higher, cancel buy")
            return None
        # from_coin_price = min(buy_price, from_coin_price)
        trade_log = self.db.start_trade_log(origin_coin, target_coin, False)

        order_quantity = self._buy_quantity(origin_symbol, target_symbol, target_balance, from_coin_price)
        self.logger.info(f"BUY QTY {order_quantity} of <{origin_symbol}>")

        # Try to buy until successful
        order = None
        order_guard = self.stream_manager.acquire_order_guard()
        while order is None:
            try:
                order = self._make_order(
                    side=Client.SIDE_BUY,
                    coinSymbol=origin_symbol + target_symbol,
                    quantity=order_quantity,
                    quote_quantity=target_balance,
                    price=from_coin_price,
                )
                self.logger.info(order)
            except BinanceAPIException as e:
                self.logger.info(e)
                time.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"Unexpected Error: {e}")

        executed_qty = float(order.get("executedQty", 0))
        if executed_qty > 0 and order["status"] == "FILLED":
            order_quantity = executed_qty  # Market buys provide QTY of actually bought asset

        trade_log.set_ordered(origin_balance, target_balance, order_quantity)

        order_guard.set_order(origin_symbol, target_symbol, int(order["orderId"]))
        order = self.wait_for_order(order["orderId"], origin_symbol, target_symbol, order_guard, order_quantity)

        if order is None:
            return None

        self.logger.info(f"Bought {origin_symbol}")

        trade_log.set_complete(order.cumulative_quote_qty)

        return order

    def sell_alt(self, origin_coin: Coin, target_coin: Coin, sell_price: float) -> BinanceOrder:
        return self.retry(self._sell_alt, origin_coin, target_coin, sell_price)

    def _sell_quantity(self, origin_symbol: str, target_symbol: str, origin_balance: float = None):
        origin_balance = origin_balance or self.get_currency_balance(origin_symbol)

        origin_tick = self.get_alt_tick(origin_symbol, target_symbol)
        return math.floor(origin_balance * 10 ** origin_tick) / float(10 ** origin_tick)

    def get_pair_orders(self, origin_symbol: str, target_symbol: str):
        """
        Query open orders for a given pair
        """

        orders = None

        try:
            if origin_symbol != target_symbol:
                orders = self.binance_client.get_open_orders(symbol=origin_symbol + target_symbol)
        except BinanceAPIException as e:
            self.logger.info(f"Unexpected Error Getting orders from {origin_symbol} to {target_symbol}")
            self.logger.info(e)
            time.sleep(1)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning(f"Unexpected Error: {e}")

        return orders

    def cancel_previous_orders(self, origin_symbol: str, target_symbol: str):
        """
        Check if there are previous orders and cancel
        """
        orders = self.get_pair_orders(origin_symbol, target_symbol)
        if orders is not None:
            for order in orders:
                cancel_order = None
                while cancel_order is None:
                    cancel_order = self.binance_client.cancel_order(
                        symbol=origin_symbol + target_symbol, orderId=order["orderId"]
                    )

    def cancel_order(self, coinSymbol: str, orderId: str):
        """
        Check if there are previous orders and cancel
        """
        cancel_order = None
        while cancel_order is None:
            cancel_order = self.binance_client.cancel_order(symbol=coinSymbol, orderId=orderId)

    def _sell_alt(self, origin_coin: Coin, target_coin: Coin, sell_price: float):
        """
        Sell altcoin
        """
        origin_symbol = origin_coin.symbol
        target_symbol = target_coin.symbol

        # get fresh balances
        with self.cache.open_balances() as balances:
            balances.clear()

        origin_balance = self.get_currency_balance(origin_symbol)
        target_balance = self.get_currency_balance(target_symbol)
        from_coin_price = self.get_sell_price(origin_symbol + target_symbol)

        sell_max_price_change = float(self.config.SELL_MAX_PRICE_CHANGE)
        if from_coin_price < sell_price * (1.0 - sell_max_price_change):
            self.logger.info("Sell price became lower, skipping sell")
            return None  # skip selling below price from ratio
        # from_coin_price = max(from_coin_price, sell_price)

        trade_log = self.db.start_trade_log(origin_coin, target_coin, True)

        order_quantity = self._sell_quantity(origin_symbol, target_symbol, origin_balance)
        self.logger.info(f"Selling {order_quantity} of {origin_symbol}")

        self.logger.info(f"Balance is {origin_balance}")
        order = None
        order_guard = self.stream_manager.acquire_order_guard()
        while order is None:
            try:
                order = self._make_order(
                    side=Client.SIDE_SELL,
                    coinSymbol=origin_symbol + target_symbol,
                    quantity=order_quantity,
                    quote_quantity=target_balance,
                    price=from_coin_price,
                )
                self.logger.info(order)
            except BinanceAPIException as e:
                self.logger.info(e)
                time.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"Unexpected Error: {e}")

        self.logger.info("order")
        self.logger.info(order)

        trade_log.set_ordered(origin_balance, target_balance, order_quantity)

        order_guard.set_order(origin_symbol, target_symbol, int(order["orderId"]))
        order = self.wait_for_order(order["orderId"], origin_symbol, target_symbol, order_guard, order_quantity)

        if order is None:
            return None

        new_balance = self.get_currency_balance(origin_symbol)
        while new_balance >= origin_balance:
            new_balance = self.get_currency_balance(origin_symbol, True)

        self.logger.info(f"Sold {origin_symbol}")

        trade_log.set_complete(order.cumulative_quote_qty)

        return order
