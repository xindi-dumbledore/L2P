import numpy as np
from LtP import *
from sklearn.ensemble import RandomForestClassifier

X_train = np.array([[0, 0, 1, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 0, 0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1, 1, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 1, 1]])
y_train = np.array([9, 56, 11, 4, 4, 6, 12])

placing = LearningToPlace(method="voting", efficient=True, clf=RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=0))
placing.fit(X_train, y_train)

X_test = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1])
predict = placing.predict(X_test)
print("Prediction: %s" % predict)
