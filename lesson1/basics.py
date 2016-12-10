# import libs
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

# import the data
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001, C=100)


x,y = digits.data[:-10], digits.target[:-10]

# training data
clf.fit(x,y)

# make prediction
print('Prediction: ', clf.predict(digits.data[-2]))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
