import backtrader as bt


class RelativeQuoteAnalyzer(bt.Analyzer):
    def stop(self):
        self.rets['relative_quote'] = self.strategy.broker.getvalue()


class TradeCountAnalyzer(bt.Analyzer):
    def start(self):
        # Initialize the counter
        self.trade_count = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            # Increment the trade count on every closed trade
            self.trade_count += 1

    def get_analysis(self):
        # Return the trade count
        return {"trade_count": self.trade_count}
