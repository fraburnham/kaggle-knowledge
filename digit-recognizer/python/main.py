import csv
from sklearn.neighbors import KNeighborsClassifier

n = KNeighborsClassifier(n_neighbors=1)
print "Loading file..."
data = list(csv.reader(open("../train.csv", "rU")))
X_train = [d[1:] for d in data[1:]]
y_train = [int(d[:1][0][0]) for d in data[1:]]
test_data = list(csv.reader(open("../test.csv", "rU")))
X_test = [d for d in test_data[1:]]
len_X_test = len(X_test)
print "Loading complete!"


print "Begin fit..."
n.fit(X_train, y_train)
print "Fit!"

print "Begin prediction..."
predictions = []
for i in range(0, len_X_test):
    predictions.append(y_train[n.kneighbors(X_test[i], return_distance=False)[0][0]])
print "Prediction finished!"

print "Output to file..."
writer = csv.writer(open("submission.csv", "w"), delimiter=",")
writer.writerow(["ImageId", "label"])
for i in range(0, len_X_test):
    writer.writerow([i+1, predictions[i]])
print "Complete!"
