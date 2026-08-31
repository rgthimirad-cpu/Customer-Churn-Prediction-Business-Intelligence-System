import pandas as pd
import joblib

model = joblib.load(
    "Machine Learning/models/final_churn_pipeline.joblib"
)

sample = pd.DataFrame({
    "Senior Citizen": [0],
    "Partner": [1],
    "Dependents": [0],
    "Tenure Months": [24],
    "Multiple Lines": [1],
    "Internet Service": ["DSL"],
    "Online Security": [1],
    "Online Backup": [1],
    "Device Protection": [1],
    "Tech Support": [1],
    "Streaming TV": [0],
    "Streaming Movies": [0],
    "Contract": ["One year"],
    "Paperless Billing": [1],
    "Payment Method": [1],
    "Monthly Charges": [70],
    "Total Charges": [1600],
    "Fiber_Monthly_Risk": [0],
    "New_High_Spend": [0],
    "New_Monthly_Customer": [0],
    "Security_Tech_Bundle": [1]
})

probability = model.predict_proba(sample)[0][1]

print("Probability:")
print(probability)