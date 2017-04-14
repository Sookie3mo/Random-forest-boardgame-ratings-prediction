from sklearn import cross_validation
import numpy as np

f = open('abalone.data', 'r')
X = []
y = []
for line in f:
	line = line.strip('\n').split(',')
	if line[0] == "M":
		line[0] = 0
	elif line[0] == "F":
		line[0] = 1
	elif line[0] == "I":
		line[0] = 2
	line_x = map(float, line[1:])
	X.append(line_x)
	y.append(line[0])
	
#print (X[0])
#print (y[0])
X = np.asarray(X)
y = np.asarray(y)

#print y[0]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y)

from forest import RandomForestClassifier
rfc_model = RandomForestClassifier( n_estimators=10, max_features=np.sqrt, max_depth=10,
                 min_samples_split=2, bootstrap=0.9)
rfc_model.fit(train_X,train_y)
accuracy = rfc_model.score(test_X,test_y)
print 'The accuracy was', accuracy, ' on the test data.'
