from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Initialize the FastAPI app
app = FastAPI()

# Load the trained models
log_model = joblib.load('../Final_Project_CCFraudDetection/logistic_regression_model.sav')
rf_model = joblib.load('../Final_Project_CCFraudDetection/random_forest_model.sav')
svm_model = joblib.load('../Final_Project_CCFraudDetection/svc_model.sav')
fnn_model = joblib.load('../Final_Project_CCFraudDetection/FNN_model.sav')

# Define the input data model
class CreditCardTransaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    scal_amount: float
    scal_time: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

# Define the prediction endpoint
@app.post("/predict")
async def predict(transaction: CreditCardTransaction):
    try:
        # Convert the transaction data to a numpy array
        transaction_data = np.array([[
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            transaction.scal_amount, transaction.scal_time
        ]])

        # Make prediction using logistic regression model
        log_prediction = int(log_model.predict(transaction_data)[0])
        log_probability = float(log_model.predict_proba(transaction_data)[:, 1][0])

        # Make prediction using random forest model
        rf_prediction = int(rf_model.predict(transaction_data)[0])
        rf_probability = float(rf_model.predict_proba(transaction_data)[:, 1][0])

        # Make prediction using SVM model
        svm_prediction = int(svm_model.predict(transaction_data)[0])
        svm_probability = float(svm_model.decision_function(transaction_data)[0])

        # Make prediction using FNN model
        fnn_prediction = int(fnn_model.predict(transaction_data)[0])
        fnn_probability = float(fnn_prediction)

        return {
            "logistic_regression": {"prediction": log_prediction, "probability": log_probability},
            "random_forest": {"prediction": rf_prediction, "probability": rf_probability},
            "svm": {"prediction": svm_prediction, "probability": svm_probability},
            "feedforward_neural_network": {"prediction": fnn_prediction, "probability": fnn_probability}
        }
    except Exception as e:
        print(e)  # Print the exception for debugging
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
