import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# ── Load ──
df = pd.read_csv('Lifestyle and survey for college going girls.csv')

# ── Clean ──
df.columns = df.columns.str.strip()

features = [
    '6. How often do you consume fast food or junk food?',
    '7. Do you consume fruits and vegetables daily?',
    '9. How often do you exercise in a week?',
    '10. Average sleep duration per night: _____ hours',
    '11. What is your primary source of stress?',
    '13. Do you experience irregular or delayed menstrual cycles?',
    '15. Do you experience acne or oily skin frequently?',
    '17. Do you experience hair fall or thinning of scalp hair?',
    '18. Have you experienced unexplained weight gain or difficulty losing weight?',
    '19. Do you feel physically tired or low in energy most of the time?'
]

target = '23. Based on medical consultation, have you ever been diagnosed with PCOS/PCOD?'

df = df[features + [target]].dropna()

# ── Convert ──
binary_cols = [
    '7. Do you consume fruits and vegetables daily?',
    '13. Do you experience irregular or delayed menstrual cycles?',
    '15. Do you experience acne or oily skin frequently?',
    '17. Do you experience hair fall or thinning of scalp hair?',
    '18. Have you experienced unexplained weight gain or difficulty losing weight?',
    '19. Do you feel physically tired or low in energy most of the time?',
    target
]

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['10. Average sleep duration per night: _____ hours'] = df[
    '10. Average sleep duration per night: _____ hours'
].replace({'5-6 hrs': 5.5, '6': 6, '7': 7, '8': 8})

df = pd.get_dummies(df, drop_first=True)
df.dropna(inplace=True)

# ── Train ──
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
print(f"{len(X.columns)} columns after encoding")

# ── Save ──
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("model.pkl saved!")
print("columns.pkl saved!")