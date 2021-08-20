import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

mu = 2.
std = 3.
a = tfp.distributions.Normal(mu, std)
print(a.rsample())
