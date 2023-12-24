import backtrader as bt


class RelativeQuoteAnalyzer(bt.Analyzer):
    def stop(self):
        self.rets['relative_quote'] = self.strategy.broker.getvalue()
