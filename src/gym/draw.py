import matplotlib.pyplot as plt
import numpy as np

rec_k5 = np.load("records_k5.npy")
rec_k3 = np.load("records_k3.npy")
rec_k10 = np.load("records_k10.npy")
number = rec_k5.shape[0]
print(number)
x_ax = np.arange(0, number, 1)
plt.plot(x_ax, rec_k3[:number], label='k=3')
plt.plot(x_ax, rec_k5, label='k=5')
plt.plot(x_ax, rec_k10[:number], label='k=10')
plt.legend()
plt.ylim(ymin=0)
plt.show()
