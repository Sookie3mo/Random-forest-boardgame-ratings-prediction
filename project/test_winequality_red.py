from sklearn import cross_validation
import numpy as np

f = open('dataset/winequality-red.csv', 'r')
line = f.readline()
X = []
y = []
for line in f:
	line = line.strip('\n').split(';')
	line_x = map(float, line[:11])
	X.append(line_x)
	y.append(int(line[11]))
	
#print (X)
#print (y[0])
X = np.asarray(X)
y = np.asarray(y)

print X
print y
train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y)


from forest import RandomForestClassifier
rfc_model = RandomForestClassifier( n_estimators=30, max_features=np.sqrt, max_depth=20,
                 min_samples_split=2, bootstrap=0.9)
rfc_model.fit(train_X,train_y)
accuracy = rfc_model.score(test_X,test_y)
print 'The accuracy was', accuracy, ' on the test data.'
