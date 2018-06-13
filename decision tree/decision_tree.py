from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

all_data = open(r'D:\study\Python Practice\decision tree\data2.csv')
reader = csv.reader(all_data)
headers = reader.next()
# print headers

featureList = []
labelList = []

for row in reader:
	labelList.append(row[len(row) - 1])
	rowDict = {}
	for i in range(1, len(row) - 1):
		rowDict[headers[i]] = row[i]
	featureList.append(rowDict)
	# print rowDict

# print featureList

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print "dummyX: " + str(dummyX)
print vec.get_feature_names()

print "labelList: " + str(labelList)

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print "dummyY: " + str(dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print "clf: " + str(clf)

with open("decision_tree2.dot", 'w') as f:
	f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)



# print 'hello'