import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')

# 1. Kaam ke columns hi rakhenge
df = df[['survived','pclass','sex','age','fare','sibsp','parch']]

# 2. Khali age ko average se bhar do
df['age'] = df['age'].fillna(df['age'].mean())

# 3. Ladki/Ladka ko number banao
df['sex'] = df['sex'].map({'male':0, 'female':1})

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Titanic Model Accuracy:", accuracy_score(y_test, pred))