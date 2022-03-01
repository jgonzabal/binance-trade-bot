from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from .base import Base
from .coin import Coin


class MarketMargins(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "current_coin_margins"
    id = Column(Integer, primary_key=True)
    coin_symbol = Column(String)
    value_history = Column(Float)
    buy = Column(Float)
    sell = Column(Float)
    datetime = Column(DateTime)

    def __init__(self, coin: Coin):
        self.coin_symbol = coin.symbol
        self.datetime = datetime.utcnow()

    def info(self):
        return {"datetime": self.datetime.isoformat(), "coin": self.coin_symbol, "buy": self.buy, "sell": self.sell}
