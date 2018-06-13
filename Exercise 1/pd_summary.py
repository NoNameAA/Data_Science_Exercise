import pandas as pd
totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

lowest_index = totals.sum(axis=1).idxmin()
print("Row with lowest total precipitation:")
print(lowest_index)

average = totals.sum(axis=0) / counts.sum(axis=0)
print("Average precipitation in each month:")
print(average)

average_daily = totals.sum(axis=1) / counts.sum(axis=1)
print("Average precipitation in each city:")
print(average_daily)
