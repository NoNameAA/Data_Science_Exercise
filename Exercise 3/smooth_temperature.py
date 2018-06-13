import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

filename = sys.argv[1]
# print(filename)
cpu_data = pd.read_csv(filename)
# print(data)
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)



lowess = sm.nonparametric.lowess
# print(cpu_data.index.values)
new_y = lowess(cpu_data['temperature'], cpu_data.index.values, frac=0.04)
plt.plot(cpu_data['timestamp'], new_y[:, 1], 'r-')


kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]
# print(kalman_data)
initial_state = kalman_data.iloc[0]
# print(initial_state)
observation_covariance = np.diag([0.5, 0.05, 0.5]) ** 2
transition_covariance = np.diag([0.05, 0.005, 0.05]) ** 2
transition = [[1.0, -1.0, 0.7], [0, 0.6, 0.03], [0, 1.3, 0.8]]
kf = KalmanFilter(observation_covariance=observation_covariance,
                  transition_covariance=transition_covariance,
                  transition_matrices=transition,
                  initial_state_mean=initial_state)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
plt.legend(['data points', 'LOESS-smoothed line', 'Kalman-smoothed line'])
plt.xlabel('Time')
plt.ylabel('Temperature')
# plt.show()
plt.savefig('cpu.svg')