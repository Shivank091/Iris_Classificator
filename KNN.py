import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('8836201-6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
target_name = "variety"
target = df[target_name]
data = df.drop(columns=[target_name])

Xtrain, Xtest,Ytrain, Ytest = train_test_split(data, target, test_size=0.2)
model = KNeighborsClassifier()
Model = model.fit(Xtrain, Ytrain)
target_predicted = model.predict(Xtest)
accuracy = model.score(Xtest, Ytest)
model_name = model.__class__.__name__
print("Using model:", model.__class__.__name__)
print(str(round(accuracy * 100, 2)) + "%")
print("Want to input your own custom datset?")
a=int(input("Enter 1 for yes,0 fo no"))
def custom_predictor(input_data):
    input_df = pd.DataFrame([input_data], columns=data.columns)
    prediction=model.predict(input_df)
    return prediction[0]
if(a):
    give1=input("Enter 4 features seperated by spaces")

    give = list(map(float, give1.split()))
    if (len(give) !=4):
        print("Bad input format")
    else:
        print(custom_predictor(give))


joblib.dump(model, "iris_knn_model.pkl")
print("Model saved as iris_knn_model.pkl")

