from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nModel Evaluation")

    print("Accuracy :", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))