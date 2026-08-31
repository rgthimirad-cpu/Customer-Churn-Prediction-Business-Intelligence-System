### **INSTALLATION GUIDE**

##### **Customer Churn Prediction System**



**Overview**



This document explains how to install and run the Customer Churn Prediction System.



The application is developed using Streamlit and integrates a trained machine learning model to predict customer churn probability, risk level, and retention recommendations.







**System Requirements**



Before running the application, ensure the following software is available:



Required Software



\- Python 3.10 or higher

\- Visual Studio Code (Recommended)

\- Web Browser (Chrome, Edge, Firefox)







**Project Structure**





**Step 1: Open Project Folder**



Open the project folder using Visual Studio Code.







**Step 2: Create a Virtual Environment (Optional)**



Open a terminal and execute:



python -m venv venv

Activate the virtual environment.

Windows

venv\\Scripts\\activate

Linux / Mac

source venv/bin/activate



**Step 3: Install Required Libraries**



Install all project dependencies using:

pip install -r requirements.txt

If the requirements file is unavailable, install manually:

pip install streamlit pandas numpy scikit-learn joblib



**Step 4: Verify Model File**



Ensure the trained machine learning model is located at:

Machine Learning/models/final\_churn\_pipeline.joblib

The application will not run if the model file is missing.



**Step 5: Run the Application**



Start the application using:

streamlit run app.py

After execution, Streamlit will automatically launch the application in a web browser.

If a browser does not open automatically, copy the generated local URL into a browser.



**Common Issues**

ModuleNotFoundError

Install missing dependencies:

pip install -r requirements.txt



**Stopping the Application**

To stop the application, return to the terminal and press:

CTRL + C



**Installation Complete**

The Customer Churn Prediction System is now ready for use.



