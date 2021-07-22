class Environment:

    def __init__(self, price_data=None, vol_data=None, ks_data=None):
        self.price_data = price_data
        self.vol_data = vol_data
        self.ks_data = ks_data
        self.observe_price = None
        self.observe_vol = None
        self.observe_ks = None
        self.idx = -1

    def reset(self):
        self.idx = -1

    def observe(self):
        if len(self.price_data) > self.idx + 1:
            self.idx += 1
            self.observe_price = self.price_data.iloc[self.idx]
            self.observe_vol = self.vol_data.iloc[self.idx]
            self.observe_ks = self.ks_data.iloc[self.idx]
            return self.observe_price
        return None

    def get_price(self):
        if self.observe_price is not None:
            return self.observe_price.values
        return None

    def get_vol(self):
        if self.observe_vol is not None:
            return self.observe_vol.values
        return None

    def get_ks(self):
        if self.observe_ks is not None:
            return self.observe_ks.values[0]
        return None

    def get_ks_to_reset(self):
        return self.ks_data.iloc[0].values[0]
