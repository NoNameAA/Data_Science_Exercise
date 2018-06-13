import pandas as pd
import numpy as np
import sys
import gzip
import math
import matplotlib.pyplot as plt


# Reference: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
# Convert javascript to python
def deg2rad(deg):
	return deg * (math.pi / 180)


# Reference: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
# Convert javascript to python
def distance(lat1, lon1, lat2, lon2):
	R = 6371
	dLat = deg2rad(lat2 - lat1)
	dLon = deg2rad(lon2 - lon1)
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	return d


def get_best_tmax(city, stations):
	stations['dis'] = np.vectorize(distance)(city.latitude, city.longitude, stations['latitude'], stations['longitude'])
	t = stations['dis'].idxmin(axis=1)
	city.best_tmax = stations.loc[t].avg_tmax
	return city


station_file = sys.argv[1]
city_file = sys.argv[2]
output_file = sys.argv[3]

# read station data
station_fh = gzip.open(station_file, 'rt', encoding='utf-8')
station_data = pd.read_json(station_fh, lines=True)

# pre-process
station_data['avg_tmax'] = station_data['avg_tmax'].apply(lambda x: x/10.0)
station_data = station_data.drop(['elevation', 'station', 'observations'], axis=1)
station_data.reset_index(inplace=True)

# read city data
city_data = pd.read_csv(city_file)
city_data.dropna(subset=['population', 'area'], inplace=True)
city_data.reset_index(inplace=True)
city_data = city_data.drop('index', axis=1)

# pre-process
city_data['area'] = city_data['area'].apply(lambda x: x/1000000.0)
city_data['density'] = city_data['population'] / city_data['area']


station_data['dis'] = pd.Series(0.0, index=station_data.index)
city_data['best_tmax'] = pd.Series(0.0, index=city_data.index)
city_data = city_data.apply(get_best_tmax, stations=station_data, axis=1)

plt.plot(city_data['best_tmax'], city_data['density'], 'b.')
plt.xlabel('Avg Max Temperature (\u00b0C)')
plt.ylabel('Population Density (people/km\u00b2)')
# plt.show()
plt.savefig(output_file)


