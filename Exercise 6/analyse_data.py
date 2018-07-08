import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

_, p_value = ss.f_oneway(data['qs1'], data['qs2'], data['qs3'], data['qs4'], data['qs5'], data['merge1'], data['partition_sort'])
print("The p-value of ANOVA: ", p_value)

print("The mean of qs1: ", round(data['qs1'].mean(), 4))
print("The mean of qs2: ", round(data['qs2'].mean(), 4))
print("The mean of qs3: ", round(data['qs3'].mean(), 4))
print("The mean of qs4: ", round(data['qs4'].mean(), 4))
print("The mean of qs5: ", round(data['qs5'].mean(), 4))
print("The mean of merge1: ", round(data['merge1'].mean(), 4))
print("The mean of partition_sort: ", round(data['partition_sort'].mean(), 4))

x_data = pd.DataFrame({'qs1': data['qs1'],
                       'qs2': data['qs2'],
                       'qs3': data['qs3'],
                       'qs4': data['qs4'],
                       'qs5': data['qs5'],
                       'merge1': data['merge1'],
                       'partition_sort': data['partition_sort']})
x_melt = pd.melt(x_data)
# print(x_melt)
posthoc = pairwise_tukeyhsd(x_melt['value'], x_melt['variable'], alpha=0.05)
print(posthoc)
plt = posthoc.plot_simultaneous()
