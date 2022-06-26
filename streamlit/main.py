import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.markdown(
    """
    <style>
         .main {
         background-color: #E5E7E9;
         }

    </style>   
    """,
    unsafe_allow_html=True
)


@st.cache
def  get_data(filename):
    taxi_data = pd.read_parquet(filename)
    return taxi_data


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title('Welcome to the project')
    st.text('Project using the NYC Taxis Datasets')

with dataset:
    st.header('NYC Taxi Dataset')
    st.text('This Dataset is available online')
    taxi_data = get_data('data/yellow_tripdata_2020-01.parquet')
    st.write(taxi_data.head())

    st.subheader('Pick up location ID')
    pulocationDist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).sort_values('PULocationID', ascending=False).head(50)
    st.bar_chart(pulocationDist)

with features:
    st.header('The feature i created')
    st.markdown('* **This feature was created because of this**')
    st.markdown('* **The second feature was created because of this and i used the following logic**')
    


with modelTraining:
    st.header('Training the model')
    st.text('Change hiperparameters of the project')

    sel_col, displ_col = st.columns(2)
    max_depth = sel_col.slider("max_depth", min_value = 10, max_value = 100, value = 20, step = 10)
    n_estimators = sel_col.selectbox("Number of estimatores", options = [100,200,300, "No Limit"], index = 0)
    sel_col.text("List of the columns in the data")
    input_feature = sel_col.selectbox("Feature to use on training", taxi_data.columns)



    if n_estimators == 'No Limit':
        n_estimators = 5000

    regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators, n_jobs = -1)

    sampleDF = taxi_data.sample(frac=0.01)
    X = sampleDF[[input_feature]]
    y = sampleDF[['trip_distance']]

    regr.fit(X,y)
    prediction = regr.predict(X)

    displ_col.subheader("MAE of model: ")
    displ_col.write(mean_absolute_error(y,prediction)) 

    displ_col.subheader("MSE of model: ")
    displ_col.write(mean_squared_error(y,prediction)) 

    displ_col.subheader("R2 score of model: ")
    displ_col.write(r2_score(y,prediction)) 