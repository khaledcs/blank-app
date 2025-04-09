import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Define sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ("Train Model", "Check Prediction"))

# ----------------------
# TRAIN MODEL PAGE
# ----------------------
if page == "Train Model":
    
    # File uploader to accept the training data CSV.
    uploaded_file = st.file_uploader("Choose a training CSV file", type="csv")

    if uploaded_file is not None:
        app_df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(app_df.head())
    
        uploaded_file1 = st.file_uploader("Choose 2 a training CSV file", type="csv")
    
        if uploaded_file1 is not None:
            credit_df = pd.read_csv(uploaded_file1)
            st.write("Data preview:")
            st.dataframe(credit_df.head())
    
            #app_df = pd.read_csv('application_record.csv')
            #credit_df = pd.read_csv('credit_record.csv')
    
            # Merge datasets on the common column "ID"
            merged_df = pd.merge(app_df, credit_df, on="ID", how="inner")
    
            # --------------------------
            # STEP 2: Create Binary Target Variable
            # --------------------------
            # The "Status" column comes from credit_df.
            # Create a binary target where "C" means approved (1) and any other value means rejected (0).
            merged_df['Target'] = np.where(merged_df['STATUS'] == 'C', 1, 0)
    
            # --------------------------
            # STEP 3: Prepare Features and Labels
            # --------------------------
            # Here we drop the original Status and target columns from the predictors.
            # You can adjust the list of columns to drop as needed.
            X = merged_df.drop(columns=['ID', 'STATUS', 'Target'])
            y = merged_df['Target']
    
            # --------------------------
            # STEP 4: Preprocess Data
            # --------------------------
            # For simplicity, use one-hot encoding on categorical predictors.
            X_processed = pd.get_dummies(X)
    
            # --------------------------
            # STEP 5: Train/Test Split and Model Training
            # --------------------------
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
            # Initialize and fit the Random Forest classifier
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
    
            # Evaluate on test data
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Test Accuracy:", accuracy)
    
            # --------------------------
            # STEP 6: Save the Model and Feature List
            # --------------------------
            # Saving the trained model and the feature columns is important so that the Streamlit app
            # can build input forms that match the training data.
            joblib.dump(model, 'rf_model.pkl')
            joblib.dump(X_processed.columns, 'model_features.pkl')
    
            print("Model and feature list saved.")



# ----------------------
# CHECK PREDICTION PAGE
# ----------------------
elif page == "Check Prediction":
    st.title("Prediction Check")
    st.write("Enter the feature values below to predict approval (Status 'C') or rejection.")

    # Try to load the trained model and feature names.
    try:
        model = joblib.load("rf_model.pkl")
        model_features = joblib.load("model_features.pkl")
    except Exception as e:
        st.error("Error: Model not found. Please train the model first on the 'Train Model' page.")
        st.stop()

    st.write("### Input the feature values")
    # Create a dictionary that will store the user input for each feature.
    user_input = {}
    # Loop through each expected feature and create an input widget.
    # For this simple example, we use text_input; in practice you might adjust input types.
    for feature in model_features:
        user_input[feature] = st.text_input(f"{feature}", value="0")
    
    # Prepare the user input for prediction.
    input_df = pd.DataFrame([user_input])
    # Convert appropriate values to numeric types.
    for col in input_df.columns:
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except ValueError:
            pass  # leave it if not convertible

    if st.button("Predict"):
        # Make sure the input DataFrame exactly matches the model features, filling missing columns with 0.
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.success("Prediction: Approved")
        else:
            st.error("Prediction: Rejected")
'''
# --------------------------
# Load Trained Model and Feature List
# --------------------------
model = joblib.load('rf_model.pkl')
model_features = joblib.load('model_features.pkl')

st.title("Credit Record Approval Prediction")
st.write("Fill in the values for the features below to predict if a record will be approved (Status 'C') or rejected.")

# --------------------------
# Generate Dynamic Input Fields for each Feature
# --------------------------
st.write("### Enter Feature Values")
# Create a dictionary to collect user inputs.
# The default value is set to "0"; you may want to customize according to the expected range or type.
user_input = {}
for feature in model_features:
    user_input[feature] = st.text_input(f"{feature}", value="0")

# Convert user inputs into a DataFrame.
input_df = pd.DataFrame([user_input])

# Try to convert user inputs to numeric types (if applicable)
for col in input_df.columns:
    try:
        input_df[col] = pd.to_numeric(input_df[col])
    except ValueError:
        pass  # leave as is if cannot convert

# --------------------------
# Prediction Button
# --------------------------
if st.button("Predict"):
    # Ensure the input DataFrame has the exact same columns as the model training set.
    # This fills in missing features with 0.
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Make prediction using the trained model
    prediction = model.predict(input_df)

    # Display prediction result
    if prediction[0] == 1:
        st.success("Prediction: Approved")
    else:
        st.error("Prediction: Rejected")





import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to Mr khaled [docs.streamlit.io](https://docs.streamlit.io/)."
)


st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)


st.title('My Dev')
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6],
    'ExamScore': [50, 55, 60, 65, 70, 75],
    'PassedTarget': [0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['StudyHours', 'ExamScore']]
y = df['PassedTarget']

log_reg_l2 = LogisticRegressionCV(penalty='l2', cv=3)  # L2 regularization with cross-validation
log_reg_l2.fit(X, y)

log_reg = LogisticRegression()
log_reg.fit(X, y)

st.write("Logistic Regression Coefficients:")
st.write(log_reg.coef_)
st.write("Intercept:" )
st.write(log_reg.intercept_)

marks=0
studytime=0
marks = st.text_input("Enter Marks", )

studytime = st.text_input("Study Time", )

#if marks>0:
new_student = np.array([[int(studytime), int(marks)]])
pred_standard = log_reg.predict(new_student)
st.write("\nStandard Logistic Regression Prediction:", "Pass" if pred_standard[0] == 1 else "Fail")

pred_l2 = log_reg_l2.predict(new_student)
st.write("L2 Regularized Logistic Regression Prediction:", "Pass" if pred_l2[0] == 1 else "Fail")


#import streamlit as st
import pyperclip
from time import sleep

def copy_to_clipboard(text):
    st.write(text)
    st.info("Text copied to clipboard!")

try:
    some_text = st.session_state['some_text']
except KeyError:
    st.write("Generating some_text...")
    sleep(5)
    some_text = "lorum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    st.session_state['some_text'] = some_text

st.markdown(some_text)

sleep(5)

if st.button(f"Copy to Clipboard"):
    copy_to_clipboard(some_text)

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

st.title("Upload Training File and Retrain Model")

# File uploader to accept the training data CSV.
uploaded_file = st.file_uploader("Choose a training CSV file", type="csv")

if uploaded_file is not None:
    app_df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(app_df.head())

    uploaded_file1 = st.file_uploader("Choose 2 a training CSV file", type="csv")

    if uploaded_file1 is not None:
        credit_df = pd.read_csv(uploaded_file1)
        st.write("Data preview:")
        st.dataframe(credit_df.head())

        #app_df = pd.read_csv('application_record.csv')
        #credit_df = pd.read_csv('credit_record.csv')

        # Merge datasets on the common column "ID"
        merged_df = pd.merge(app_df, credit_df, on="ID", how="inner")

        # --------------------------
        # STEP 2: Create Binary Target Variable
        # --------------------------
        # The "Status" column comes from credit_df.
        # Create a binary target where "C" means approved (1) and any other value means rejected (0).
        merged_df['Target'] = np.where(merged_df['STATUS'] == 'C', 1, 0)

        # --------------------------
        # STEP 3: Prepare Features and Labels
        # --------------------------
        # Here we drop the original Status and target columns from the predictors.
        # You can adjust the list of columns to drop as needed.
        X = merged_df.drop(columns=['ID', 'STATUS', 'Target'])
        y = merged_df['Target']

        # --------------------------
        # STEP 4: Preprocess Data
        # --------------------------
        # For simplicity, use one-hot encoding on categorical predictors.
        X_processed = pd.get_dummies(X)

        # --------------------------
        # STEP 5: Train/Test Split and Model Training
        # --------------------------
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Initialize and fit the Random Forest classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy:", accuracy)

        # --------------------------
        # STEP 6: Save the Model and Feature List
        # --------------------------
        # Saving the trained model and the feature columns is important so that the Streamlit app
        # can build input forms that match the training data.
        joblib.dump(model, 'rf_model.pkl')
        joblib.dump(X_processed.columns, 'model_features.pkl')

        print("Model and feature list saved.")
'''
