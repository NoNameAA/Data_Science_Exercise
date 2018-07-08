import time
from implementations import all_implementations
import numpy as np
import pandas as pd

data = pd.DataFrame(columns=['sort_name', 'sort_time'])

row = 0
times = 50


for _ in range(times):
	arr = np.random.randint(10000, size=10000)
	for sort in all_implementations:
		st = time.time()
		res = sort(arr)
		en = time.time()

		data.loc[row] = [sort.__name__, round(en-st, 4)]
		row += 1


results = data['sort_time'].values
results = results.reshape(times, 7)
# print(results)


open('data.csv', 'w').close()

f = open('data.csv', 'a')
s = ""
for sort in all_implementations:
	s = s + sort.__name__ + ','
s = s[:-1]+'\n'

f.write(s)

np.savetxt(f, results, delimiter=',', fmt='%.4f')
f.close()
