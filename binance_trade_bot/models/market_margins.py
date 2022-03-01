from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base
from .coin import Coin


class MarketMargins(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "current_coin_margins"
    id = Column(Integer, primary_key=True)
    coin_id = Column(String, ForeignKey("coins.symbol"))
    coin = relationship("Coin", foreign_keys=[coin_id], lazy="joined")
    value_history = Column(Float)
    buy = Column(Float)
    sell = Column(Float)
    datetime = Column(DateTime)

    def __init__(self, coin: Coin):
        self.coin = coin
        self.datetime = datetime.utcnow()

    def info(self):
        return {"datetime": self.datetime.isoformat(), "coin": self.coin.info(), "buy": self.buy, "sell": self.sell}
