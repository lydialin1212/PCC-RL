import matplotlib.pyplot as plt
import numpy as np

rec = np.load("records.npy")
number = rec.shape[0]
print(number)
x_ax = np.arange(0, number, 1)
plt.plot(x_ax, rec)
plt.ylim(ymin=0)
plt.show()
