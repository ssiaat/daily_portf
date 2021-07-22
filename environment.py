class Environment:

    def __init__(self, price_data=None, cap_data=None, ks_data=None):
        self.price_data = price_data
        self.cap_data = cap_data
        self.ks_data = ks_data
        self.observe_price = None
        self.observe_cap = None
        self.observe_ks = None
        self.idx = -1

    def reset(self):
        self.idx = -1

    def observe(self):
        if len(self.price_data) > self.idx + 1:
            self.idx += 1
            self.observe_price = self.price_data.iloc[self.idx]
            self.observe_cap = self.cap_data.iloc[self.idx]
            self.observe_ks = self.ks_data.iloc[self.idx]
            return self.observe_price
        return None

    def get_price(self):
        if self.observe_price is not None:
            return self.observe_price.values
        return None

    def get_cap(self):
        if self.observe_cap is not None:
            return self.observe_cap.values
        return None

    def get_ks(self):
        if self.observe_ks is not None:
            return self.observe_ks.values[0]
        return None

    def get_ks_to_reset(self):
        return self.ks_data.iloc[0].values[0]
