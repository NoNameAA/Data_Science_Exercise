import pandas as pd
import numpy as np
import sys
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

labelled_file = sys.argv[1]
unlabelled_file = sys.argv[2]

label_data = pd.read_csv(labelled_file)
unlabel_data = pd.read_csv(unlabelled_file)

y = label_data['city']
X = label_data
X = X.drop(['city', 'year'], axis=1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

x_predict = unlabel_data
x_predict = scaler.transform(x_predict.drop(['city', 'year'], axis=1))

svc_model = SVC(kernel='linear', C=0.3)
svc_model.fit(x_train, y_train)
score = svc_model.score(x_test, y_test)
prediction = svc_model.predict(x_predict)

df = pd.DataFrame({'truth': y_test, 'prediction': svc_model.predict(x_test)})
print(df[df['truth'] != df['prediction']])

# knn_model = KNeighborsClassifier(n_neighbors=30)
# knn_model.fit(x_train, y_train)
# prediction = knn_model.predict(x_predict)
# score = knn_model.score(x_test, y_test)
# print(prediction)
print("The score of model is: ", score)
pd.Series(prediction).to_csv(sys.argv[3], index=False)









