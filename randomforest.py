from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Evaluation for Random Forest
print("Train Accuracy : {:.2f} %".format(accuracy_score(rf_model.predict(X_train), Y_train) * 100))
print("Test Accuracy  : {:.2f} %".format(accuracy_score(rf_model.predict(X_test), Y_test) * 100))
print("Precision      : {:.2f} %".format(precision_score(rf_model.predict(X_test), Y_test, average='micro') * 100))
print("Recall         : {:.2f} %".format(recall_score(rf_model.predict(X_test), Y_test, average='micro') * 100))
