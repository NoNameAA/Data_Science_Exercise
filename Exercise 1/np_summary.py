import numpy as np
data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

total_precipitation = totals.sum(axis = 1)
lowest = total_precipitation.argmin()
print("Row with lowest total precipitation:\n",lowest)

total_precipitation = totals.sum(axis = 0)
sum_counts = counts.sum(axis = 0)
print("Average precipitation in each month:\n", total_precipitation / sum_counts)

total_precipitation = totals.sum(axis = 1)
sum_counts = counts.sum(axis = 1)
print("Average precipitation in each city:\n", total_precipitation / sum_counts)

city_num = int(totals.size / 12)
new_totals = totals.reshape(int(totals.size / 3), 3)
sum = new_totals.sum(axis = 1)
result_sum = sum.reshape(city_num, 4)
print("Quarterly precipitation totals:\n", result_sum)