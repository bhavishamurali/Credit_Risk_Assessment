import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Setup output folders
os.makedirs('outputs', exist_ok=True)

print("🚀 Synthesizing Local Financial Applicant Dataset...")
np.random.seed(42)
num_applicants = 500

# Generate simulated banking metadata
data = {
    'Credit_Score': np.random.randint(300, 850, size=num_applicants),
    'Annual_Income_K': np.random.randint(20, 150, size=num_applicants),
    'Debt_to_Income_Ratio': np.random.uniform(0.1, 0.8, size=num_applicants),
    'Employment_Years': np.random.randint(0, 20, size=num_applicants)
}

df = pd.DataFrame(data)

# Formulate a baseline default logic constraint rule
# Lower credit score + higher debt-to-income ratio = high risk of default (1)
df['Default'] = np.where(
    (df['Credit_Score'] < 580) & (df['Debt_to_Income_Ratio'] > 0.4) | (df['Credit_Score'] < 450), 
    1, 0
    
)

# Split features and labels
X = df.drop(columns=['Default'])
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest Credit Assessment Classifier...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Calculate testing threshold accuracy metrics
accuracy = model.score(X_test, y_test)
print(f"✅ Baseline Model Training Accuracy: {accuracy * 100:.2f}%")

print("💾 Exporting saved model matrix weights...")
with open('outputs/credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 Engine ready! Run 'python -m streamlit run app.py' next.")