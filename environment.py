class Environment:

    def __init__(self, price_data=None, vol_data=None):
        self.price_data = price_data
        self.vol_data = vol_data
        self.observ_price = price_data
        self.observ_vol = vol_data
        self.idx = -1

    def reset(self):
        self.idx = -1

    def observe(self):
        if len(self.price_data) > self.idx + 1:
            self.idx += 1
            self.observ_price = self.price_data.iloc[self.idx]
            self.observ_vol = self.vol_data.iloc[self.idx]
            return self.observ_price
        return None

    def get_price(self):
        if self.observ_price is not None:
            return self.observ_price
        return None

    def get_vol(self):
        if self.observ_vol is not None:
            return self.observ_vol
        return None
