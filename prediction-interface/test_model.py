import joblib

model = joblib.load(
    "Machine Learning/models/final_churn_pipeline.joblib"
)

print(type(model))
print(model)