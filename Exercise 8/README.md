This exercise is for analyzing the differences between GaussianNB-based classifiers, k-nearest neighbours classifier and SVM classifier.

GaussianNB-based classifiers:
>> bayes_rgb_model = GaussianNB(priors=None)

k-nearest neighbours classifier:
>> knn_rgb_model = KNeighborsClassifier(n_neighbors=9)

SVM classifier:
>> svc_rgb_model = SVC(kernel='linear', C=3)
