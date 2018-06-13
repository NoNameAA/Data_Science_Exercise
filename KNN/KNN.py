from sklearn import neighbors
from sklearn import datasets
import csv
import random
import math
import operator
import matplotlib.pyplot as plt


def split_data(iris_data, split_rate, train_data, test_data):
	for i in range(0, len(iris_data)):
		random_number = random.random()
		for j in range(0, len(iris_data[i])-1):
			iris_data[i][j] = float(iris_data[i][j])
		if random_number < split_rate:
			train_data.append(iris_data[i])
		else:
			test_data.append(iris_data[i])


def get_distance(data_x, data_y):
	dis = 0
	for i in range(0, len(data_x)-1):
		dis += math.sqrt(pow(data_x[i] - data_y[i], 2))
	return dis


def heap_maintain(heap, k):
	size = len(heap) - 1
	if size <= k:
		current = size
		while current/2 >= 1:
			if heap[current][1] > heap[current/2][1]:
				t = heap[current]
				heap[current] = heap[current/2]
				heap[current/2] = t
			else:
				break
			current /= 2
	else:
		if heap[-1][1] >= heap[1][1]:
			return
		heap[1] = heap[-1]
		current = 1
		while current*2 <= k:
			max_index = current*2
			if max_index + 1 <= k and heap[max_index][1] < heap[max_index + 1][1]:
				max_index = max_index + 1
			if heap[max_index][1] > heap[current][1]:
				t = heap[current]
				heap[current] = heap[max_index]
				heap[max_index] = t
				current = max_index
			else:
				break


def KNN_predict(train_data, test_data, k, predict_result):
	for i in range(0, len(test_data)):
		heap = [(0, 0)]
		for j in range(0, len(train_data)):
			distance = get_distance(train_data[j], test_data[i])
			heap.append((j, distance))
			heap_maintain(heap, k)
		# print ""
		# for tt in range(1, k+1):
		# 	print heap[tt][1],
		counter = {}
		for j in range(1, k+1):
			iris_index = heap[j][0]
			if train_data[iris_index][-1] in counter:
				counter[train_data[iris_index][-1]] += 1
			else:
				counter[train_data[iris_index][-1]] = 1
		sorted_counter = sorted(counter.items(), key=operator.itemgetter(1))
		predict_result.append(sorted_counter[-1][0])
		# print predict_result[-1]


def get_accuracy(test_data, predict_result):
	size = len(test_data)
	correct_number = 0
	wrong_number = 0

	for i in range(0, size):
		print ">> actual data: ", test_data[i][-1], "| predict data: ", predict_result[i]
		if test_data[i][-1] != predict_result[i]:
			wrong_number += 1
		else:
			correct_number += 1

	accuracy = float(correct_number) / size

	print "Total number: ", size
	print "Correct number: ", correct_number
	print "Wrong number: ", wrong_number
	print "Accuracy: ", accuracy
	plt.plot(0.5, accuracy, 'ro')
	plt.axis([0, 1, 0, 1])
	plt.show()

if __name__ == "__main__":
	iris_file = open("./iris_data.csv")
	iris_reader = csv.reader(iris_file)
	iris_data = list(iris_reader)

	train_data = []
	test_data = []
	predict_result = []

	split_data(iris_data, 0.5, train_data, test_data)
	KNN_predict(train_data, test_data, 7, predict_result)
	get_accuracy(test_data, predict_result)









