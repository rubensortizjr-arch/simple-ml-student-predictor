from sklearn.linear_model import LogisticRegression

# Training data (grades)
X = [[6], [7], [8], [4], [5], [9]]
y = [0, 1, 1, 0, 0, 1]  # 0 = Failed, 1 = Approved

model = LogisticRegression()
model.fit(X, y)

grade = float(input("Enter student grade: "))
result = model.predict([[grade]])

if result == 1:
    print("Prediction: Approved ✅")
else:
    print("Prediction: Failed ❌")
