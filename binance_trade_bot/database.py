import json
import os
import time
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from multiprocessing.dummy import Array
from typing import List, Optional, Union

from socketio import Client
from socketio.exceptions import ConnectionError as SocketIOConnectionError
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from .config import Config
from .logger import Logger
from .models import *  # pylint: disable=wildcard-import,unused-wildcard-import

LogScout = namedtuple("LogScout", ["pair", "target_ratio", "coin_price", "optional_coin_price"])


class Database:
    def __init__(self, logger: Logger, config: Config, uri="sqlite:///data/crypto_trading.db", isTest=False):
        self.logger = logger
        self.config = config
        self.engine = create_engine(uri)
        self.session_factory = scoped_session(sessionmaker(bind=self.engine))
        self.socketio_client = Client()
        self.isTest = isTest

    def socketio_connect(self):
        if self.isTest:
            return False
        if self.socketio_client.connected and self.socketio_client.namespaces:
            return True
        try:
            if not self.socketio_client.connected:
                self.socketio_client.connect("http://api:5123", namespaces=["/backend"])
            while not self.socketio_client.connected or not self.socketio_client.namespaces:
                time.sleep(0.1)
            return True
        except SocketIOConnectionError:
            return False

    @contextmanager
    def db_session(self):
        """
        Creates a context with an open SQLAlchemy session.
        """
        session: Session = self.session_factory()
        yield session
        session.commit()
        session.close()

    def set_coins(self, symbols: List[str]):
        session: Session

        # Add coins to the database and set them as enabled or not
        with self.db_session() as session:
            # For all the coins in the database, if the symbol no longer appears
            # in the config file, set the coin as disabled
            coins: List[Coin] = session.query(Coin).all()
            for coin in coins:
                if coin.symbol not in symbols:
                    coin.enabled = False

            # For all the symbols in the config file, add them to the database
            # if they don't exist
            for symbol in symbols:
                coin = next((coin for coin in coins if coin.symbol == symbol), None)
                if coin is None:
                    session.add(Coin(symbol))
                else:
                    coin.enabled = True

        # For all the combinations of coins in the database, add a pair to the database
        with self.db_session() as session:
            coins: List[Coin] = session.query(Coin).filter(Coin.enabled).all()
            for from_coin in coins:
                for to_coin in coins:
                    if from_coin != to_coin:
                        pair = session.query(Pair).filter(Pair.from_coin == from_coin, Pair.to_coin == to_coin).first()
                        if pair is None:
                            session.add(Pair(from_coin, to_coin))

    def get_coins(self, only_enabled=True) -> List[Coin]:
        session: Session
        with self.db_session() as session:
            if only_enabled:
                coins = session.query(Coin).filter(Coin.enabled).all()
            else:
                coins = session.query(Coin).all()
            session.expunge_all()
            return coins

    def get_coin(self, coin: Union[Coin, str]) -> Coin:
        if isinstance(coin, Coin):
            return coin
        session: Session
        with self.db_session() as session:
            coin = session.query(Coin).get(coin)
            session.expunge(coin)
            return coin

    def set_current_coin(self, coin: Union[Coin, str]):
        coin = self.get_coin(coin)
        session: Session
        with self.db_session() as session:
            if isinstance(coin, Coin):
                coin = session.merge(coin)
            cc = CurrentCoin(coin)
            session.add(cc)
            self.send_update(cc)

    def get_current_coin(self) -> Optional[Coin]:
        session: Session
        with self.db_session() as session:
            current_coin = session.query(CurrentCoin).order_by(CurrentCoin.datetime.desc()).first()
            if current_coin is None:
                return None
            coin = current_coin.coin
            session.expunge(coin)
            return coin

    def set_current_margins(self, coin: Union[Coin, str], value: float, buy: float, sell: float):
        coin = self.get_coin(coin)
        session: Session
        with self.db_session() as session:
            if isinstance(coin, Coin):
                coin = session.merge(coin)

            mm = MarketMargins(coin)
            mm.sell = sell
            mm.buy = buy
            mm.value_history = value
            session.add(mm)
            self.send_update(mm)

    def get_current_margins(self) -> Union[Coin, float, float]:
        session: Session
        with self.db_session() as session:
            current_coin_margins = session.query(MarketMargins).order_by(MarketMargins.datetime.desc()).first()
            if current_coin_margins is None:
                return [None, None, None]
            coin = current_coin_margins.coin_symbol
            buy = current_coin_margins.buy
            sell = current_coin_margins.sell
            session.expunge(current_coin_margins)
            return [coin, buy, sell]

    def get_current_history(self, coin: str) -> Array:
        session: Session
        with self.db_session() as session:
            current_coin_margins = (
                session.query(MarketMargins)
                .filter(MarketMargins.coin_symbol == coin)
                .order_by(MarketMargins.datetime.desc())
                .limit(500)
            )
            market_history = []
            for ccm in current_coin_margins:
                market_history.insert(0, ccm.value_history)
            session.expunge_all()
            return market_history

    def get_pair(self, from_coin: Union[Coin, str], to_coin: Union[Coin, str]):
        from_coin = self.get_coin(from_coin)
        to_coin = self.get_coin(to_coin)
        session: Session
        with self.db_session() as session:
            pair: Pair = session.query(Pair).filter(Pair.from_coin == from_coin, Pair.to_coin == to_coin).first()
            session.expunge(pair)
            return pair

    def get_pairs_from(self, from_coin: Union[Coin, str], only_enabled=True) -> List[Pair]:
        from_coin = self.get_coin(from_coin)
        session: Session
        with self.db_session() as session:
            pairs = session.query(Pair).filter(Pair.from_coin == from_coin)
            if only_enabled:
                pairs = pairs.filter(Pair.enabled.is_(True))
            pairs = pairs.all()
            session.expunge_all()
            return pairs

    def get_pairs(self, only_enabled=True) -> List[Pair]:
        session: Session
        with self.db_session() as session:
            pairs = session.query(Pair)
            if only_enabled:
                pairs = pairs.filter(Pair.enabled.is_(True))
            pairs = pairs.all()
            session.expunge_all()
            return pairs

    def batch_log_scout(self, logs: List[LogScout]):
        session: Session
        with self.db_session() as session:
            dt = datetime.now()
            session.execute(
                insert(ScoutHistory),
                [
                    {
                        "pair_id": ls.pair.id,
                        "target_ratio": ls.target_ratio,
                        "current_coin_price": ls.coin_price,
                        "other_coin_price": ls.optional_coin_price,
                        "datetime": dt,
                    }
                    for ls in logs
                ],
            )

    def log_scout(
        self,
        pair: Pair,
        target_ratio: float,
        current_coin_price: float,
        other_coin_price: float,
    ):
        session: Session
        with self.db_session() as session:
            pair = session.merge(pair)
            sh = ScoutHistory(pair, target_ratio, current_coin_price, other_coin_price)
            session.add(sh)
            self.send_update(sh)

    def prune_scout_history(self):
        time_diff = datetime.now() - timedelta(hours=self.config.SCOUT_HISTORY_PRUNE_TIME)
        session: Session
        with self.db_session() as session:
            session.query(ScoutHistory).filter(ScoutHistory.datetime < time_diff).delete()

    def prune_market_history(self):
        time_diff = datetime.now() - timedelta(hours=24 * self.config.SCOUT_HISTORY_PRUNE_TIME)
        session: Session
        with self.db_session() as session:
            session.query(MarketMargins).filter(MarketMargins.datetime < time_diff).delete()

    def prune_value_history(self):
        session: Session
        with self.db_session() as session:
            session.query(CoinValue).delete()

    def create_database(self):
        Base.metadata.create_all(self.engine)

    def start_trade_log(self, from_coin: Coin, to_coin: Coin, selling: bool):
        return TradeLog(self, from_coin, to_coin, selling)

    def send_update(self, model):
        if not self.socketio_connect():
            return

        self.socketio_client.emit(
            "update",
            {"table": model.__tablename__, "data": model.info()},
            namespace="/backend",
        )

    def migrate_old_state(self):
        """
        For migrating from old dotfile format to SQL db. This method should be removed in
        the future.
        """
        if os.path.isfile(".current_coin"):
            with open(".current_coin") as f:
                coin = f.read().strip()
                self.logger.info(f".current_coin file found, loading current coin {coin}")
                self.set_current_coin(coin)
            os.rename(".current_coin", ".current_coin.old")
            self.logger.info(".current_coin renamed to .current_coin.old - You can now delete this file")

        if os.path.isfile(".current_coin_table"):
            with open(".current_coin_table") as f:
                self.logger.info(".current_coin_table file found, loading into database")
                table: dict = json.load(f)
                session: Session
                with self.db_session() as session:
                    for from_coin, to_coin_dict in table.items():
                        for to_coin, ratio in to_coin_dict.items():
                            if from_coin == to_coin:
                                continue
                            pair = session.merge(self.get_pair(from_coin, to_coin))
                            pair.ratio = ratio
                            session.add(pair)

            os.rename(".current_coin_table", ".current_coin_table.old")
            self.logger.info(".current_coin_table renamed to .current_coin_table.old - " "You can now delete this file")

    def batch_update_coin_values(self, cv_batch: List[CoinValue]):
        session: Session
        with self.db_session() as session:
            session.execute(
                insert(CoinValue),
                [
                    {
                        "coin_id": cv.coin.symbol,
                        "balance": cv.balance,
                        "usd_price": cv.usd_price,
                        "btc_price": cv.btc_price,
                        "interval": cv.interval,
                        "datetime": cv.datetime,
                    }
                    for cv in cv_batch
                ],
            )


class TradeLog:
    def __init__(self, db: Database, from_coin: Coin, to_coin: Coin, selling: bool):
        self.db = db
        session: Session
        with self.db.db_session() as session:
            from_coin = session.merge(from_coin)
            to_coin = session.merge(to_coin)
            self.trade = Trade(from_coin, to_coin, selling)
            session.add(self.trade)
            # Flush so that SQLAlchemy fills in the id column
            session.flush()
            self.db.send_update(self.trade)

    def set_ordered(self, alt_starting_balance, crypto_starting_balance, alt_trade_amount):
        session: Session
        with self.db.db_session() as session:
            trade: Trade = session.merge(self.trade)
            trade.alt_starting_balance = alt_starting_balance
            trade.alt_trade_amount = alt_trade_amount
            trade.crypto_starting_balance = crypto_starting_balance
            trade.state = TradeState.ORDERED
            self.db.send_update(trade)

    def set_complete(self, crypto_trade_amount):
        session: Session
        with self.db.db_session() as session:
            trade: Trade = session.merge(self.trade)
            trade.crypto_trade_amount = crypto_trade_amount
            trade.state = TradeState.COMPLETE
            self.db.send_update(trade)


if __name__ == "__main__":
    database = Database(Logger(), Config())
    database.create_database()
