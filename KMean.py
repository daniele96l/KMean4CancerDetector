from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()
#print(bc)

#we will see something like data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
#in a numpy array
#and the lables array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,

x = scale(bc.data)
y = bc.target

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(x_train)

predictions = model.predict(x_test)

labels = model.labels_

print('labels',labels)
print('predictions',predictions)
print('accuracy',accuracy_score(y_test,predictions))
print('actual',y_test)