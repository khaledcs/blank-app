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
#Using random see 42 for reproducibility
np.random.seed(42)
mean=170
standard_deviation=10
minimum_value=140
maximum_value=220
nubmer_of_sample_individual=50
random_sample = np.random.normal(mean, standard_deviation, nubmer_of_sample_individual)
final_sample = random_sample[(random_sample >= minimum_value) & (random_sample <= maximum_value)]

final_sample= final_sample[:nubmer_of_sample_individual].tolist()
st.subheader('No of Sample: %s:00' %len(final_sample))
st.subheader ('\nSample Data: %s:00' %final_sample)



data = {
    'StudyHours': [1, 2, 3, 4, 5, 6],
    'ExamScore': [50, 55, 60, 65, 70, 75],
    'PassedTarget': [0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['StudyHours', 'ExamScore']]
y = df['PassedTarget']

log_reg = LogisticRegression()
log_reg.fit(X, y)

st.write("Logistic Regression Coefficients:")
st.write(log_reg.coef_)
st.write("Intercept:" )
st.write(log_reg.intercept_)
