from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target

# print(iris_x[:2,:])
# print(iris_y)


X_train,X_test,y_train,y_test = train_test_split(iris_x,iris_y,test_size=0.7)
# print(y_train)

# begin to classifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)



print(knn.predict(X_test))
print(' ')
print('*********************  origin Y  *********************')
print(' ')

print(y_test)
