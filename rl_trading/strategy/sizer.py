import backtrader as bt


class AllInSizer(bt.Sizer):
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            # Get commission and slippage info
            commission_info = self.broker.getcommissioninfo(data)
            commission_rate = commission_info.p.commission
            slippage_perc = self.broker.p.slip_perc

            # Estimate transaction cost and slippage
            price = data.close[0]
            gross_cost = cash / price
            commission = commission_rate * gross_cost
            slippage = slippage_perc * price

            # Adjust size to account for estimated commission and slippage
            net_cash = cash - (commission + slippage * gross_cost)
            size = net_cash / price
            return size * 0.9
        else:
            # For selling, return the size of the current position
            return self.broker.getposition(data).size
