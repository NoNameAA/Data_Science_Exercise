import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename1 = sys.argv[1]
filename2 = sys.argv[2]

table_1 = pd.read_table(filename1, sep=' ', header=None, index_col=1,
                        names=['lang', 'page', 'views', 'bytes'])

table_2 = pd.read_table(filename2, sep=' ', header=None, index_col=1,
                        names=['lang', 'page', 'views_2', 'bytes'])

sorted_table_1 = table_1.sort_values(by=['views'], ascending=False)

# print(table_1)
# new = pd.merge(table_1, table_2, how='left')
# new['views_2'] = new['views_2'].fillna(0.0).astype('int64')
# print(new[['views', 'views_2']])

table_1['views_2'] = table_2['views_2']
table_1['views_2'] = table_1['views_2'].fillna(0.0).astype('int64')

# print(table_1[['views', 'views_2']])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(sorted_table_1['views'].values, 'b-')
plt.xlabel('rank')
plt.ylabel('views')

plt.subplot(1, 2, 2)
plt.plot(table_1['views'].values, table_1['views_2'].values, 'ro')
plt.xlabel('views of first day')
plt.ylabel('views of second day')
plt.xscale('log')
plt.yscale('log')
# plt.show()
plt.savefig('wikipedia.png')


# print(table['bytes'].head())
#
# plt.figure(figsize=(10, 5)) # change the size to something sensible
# plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
# plt.plot() # build plot 1
# plt.subplot(1, 2, 2) # ... and then select the second
# plt.plot() # build plot 2
# plt.show()

