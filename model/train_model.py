import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../dataset/machine_data.csv')

X = data[['Temperature','Vibration','RunningHours']]
y = data['Status']

model = DecisionTreeClassifier()
model.fit(X,y)

prediction = model.predict([[75,0.9,500]])

print("Predicted Status:", prediction[0])
