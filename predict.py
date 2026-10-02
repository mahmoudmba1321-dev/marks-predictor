import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[["hours"]]   # input
y = data["marks"]     # what we want to predict

model = LinearRegression()
model.fit(X, y)       # the model "learns" here

# Predict for a new student
hours = 6.5
predicted = model.predict(pd.DataFrame({"hours": [hours]}))
print(f"If a student studies {hours} hours, predicted marks: {predicted[0]:.1f}")

# Draw the data and the line the model learned
plt.scatter(X, y, label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Model line")
plt.xlabel("Hours studied")
plt.ylabel("Marks")
plt.legend()
plt.show()
from sklearn.metrics import r2_score, mean_absolute_error

pred = model.predict(X)
print("R2 score:", round(r2_score(y, pred), 3))
print("Average error (marks):", round(mean_absolute_error(y, pred), 2))

# Let the user try it
user_hours = float(input("Enter hours studied: "))
result = model.predict(pd.DataFrame({"hours": [user_hours]}))
print(f"Predicted marks: {result[0]:.1f}")