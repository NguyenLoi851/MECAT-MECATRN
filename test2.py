import numpy as np
import random

selected = np.array(random.sample(range(10),9))
selected = np.sort(selected)
print(selected)