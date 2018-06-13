import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET
import math


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


def total_distance(data):
	data2 = data.shift(1)
	data['lat2'] = data2['lat']
	data['lon2'] = data2['lon']
	# print(data.dtypes)
	data['dis'] = np.vectorize(distance)(data['lat'], data['lon'], data['lat2'], data['lon2'])
	return data['dis'].sum()*1000


def read_gpx(filename):
	tree = ET.parse(filename)
	root = tree.getroot()
	lon = []
	lat = []
	for elem in root.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
		lon.append(elem.attrib['lon'])
		lat.append(elem.attrib['lat'])

	d = {'lon': lon, 'lat': lat}
	data = pd.DataFrame(data=d)
	data['lon'] = data['lon'].astype(str).astype(float)
	data['lat'] = data['lat'].astype(str).astype(float)
	return data


def output_gpx(points, output_filename):
	"""
	Output a GPX file with latitude and longitude from the points DataFrame.
	"""
	from xml.dom.minidom import getDOMImplementation
	def append_trkpt(pt, trkseg, doc):
		trkpt = doc.createElement('trkpt')
		trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
		trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
		trkseg.appendChild(trkpt)

	doc = getDOMImplementation().createDocument(None, 'gpx', None)
	trk = doc.createElement('trk')
	doc.documentElement.appendChild(trk)
	trkseg = doc.createElement('trkseg')
	trk.appendChild(trkseg)

	points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

	with open(output_filename, 'w') as fh:
		doc.writexml(fh, indent=' ')


points = read_gpx(sys.argv[1])
copy_points = points.copy()
# print('Unfiltered distance: %0.2f' % (total_distance(copy_points)))

initial_state = points.iloc[0]
# print(initial_state)
observation_covariance = np.diag([0.8, 0.8]) ** 2
transition_covariance = np.diag([0.5, 0.5]) ** 2
transition = [[1.0, 0], [0, 1.0]]
kf = KalmanFilter(observation_covariance=observation_covariance,
                  transition_covariance=transition_covariance,
                  transition_matrices=transition,
                  initial_state_mean=initial_state)


smoothed_points, _ = kf.smooth(points)
data_smoothed = pd.DataFrame(smoothed_points)
data_smoothed = pd.DataFrame({'lat': data_smoothed[0], 'lon': data_smoothed[1]})

data_smoothed_copy = data_smoothed.copy()
# print('Filtered distance: %0.2f' % (total_distance(data_smoothed_copy)))
output_gpx(data_smoothed, 'out.gpx')

with open('calc_distance.txt', 'w') as out:
	out.write('Unfiltered distance: %0.2f\n' % (total_distance(copy_points)))
	out.write('Filtered distance: %0.2f\n' % (total_distance(data_smoothed_copy)))






