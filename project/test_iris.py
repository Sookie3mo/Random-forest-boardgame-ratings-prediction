from sklearn import cross_validation
import numpy as np

f = open('dataset/iris.data', 'r')
X = []
y = []
for line in f:
	line = line.strip('\n').split(',')
	if len(line) != 5:
		continue
	if line[4] == "Iris-setosa":
		line[4] = 0
	elif line[4] == "Iris-versicolor":
		line[4] = 1
	elif line[4] == "Iris-virginica":
		line[4] = 2
	line_x = map(float, line[:4])
	X.append(line_x)
	y.append(line[4])
	
#print (X)
#print (y[0])
X = np.asarray(X)
y = np.asarray(y)

#print y
train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y)


from forest import RandomForestClassifier
rfc_model = RandomForestClassifier( n_estimators=10, max_features=np.sqrt, max_depth=10,
                 min_samples_split=2, bootstrap=0.9)
rfc_model.fit(train_X,train_y)
accuracy = rfc_model.score(test_X,test_y)
print 'The accuracy was', accuracy, ' on the test data.'
