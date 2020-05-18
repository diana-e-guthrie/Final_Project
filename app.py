import altair as alt
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

neighborhoods = ['Belcaro',
'Capitol Hill',
'Cheesman Park',
'Cherry Creek',
'City Park West',
'Cole',
'Congress Park',
'Five Points',
'Hale',
'Park Hill',
'Speer',
'Washington Park',
'Washington Virginia Vale',
]

@st.cache(allow_output_mutation=True)
def get_hood(neighborhood):
    df=pd.read_csv('https://raw.githubusercontent.com/Samuel-Rotbart/Final_Project/master/denver.hoods.csv')
    row = df.loc[df['City']==neighborhood]
    return row

@st.cache(allow_output_mutation=True)
def get_rentals():
    return pd.read_csv('https://raw.githubusercontent.com/Samuel-Rotbart/Final_Project/master/excelview.csv')

@st.cache(allow_output_mutation=True)
def clean_rentals(df):
    df.dropna(inplace = True)
    return df[["Rent", "Beds_Max","Baths_Min","Sqft_Min","Sqft_Max", "Walk Score", "Transit Score", "Bike Score"]]

@st.cache(allow_output_mutation=True)    
def train_model(df):
    X = df[["Beds_Max","Baths_Min","Sqft_Min","Sqft_Max", "Walk Score", "Transit Score", "Bike Score"]]
    y = df["Rent"].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def run_model(model, df):
    return model.predict(input_df)

def main():
    st.sidebar.subheader("DenverRents")
    hood_selection = st.sidebar.selectbox("Select Neighborhood", neighborhoods)
    beds = st.sidebar.slider("Number of Bedrooms", min_value=0, max_value=6)
    baths = st.sidebar.slider("Number of Bathrooms", min_value=1.0, max_value=6.0, step = .25)
    sq_ft = st.sidebar.number_input("Square Footage - Living Space", min_value=0.0, max_value=5000.0)
    lot = st.sidebar.number_input("Square Footage - Lot", min_value=0.0, max_value=10000.0)
    row = get_hood(hood_selection)
    walk_score = row['Walk Score'].values[0]
    transit_score = row['Transit Score'].values[0]
    bike_score = row['Bike Score'].values[0]
    input_list = [[beds, baths, sq_ft, lot, walk_score, transit_score, bike_score]]
    input_df = pd.DataFrame(input_list, columns =['beds', 'baths', 'sq_ft', 'lot', 'walk_score', 'transit_score', 'bike_score']) 
    df = get_rentals()
    df = clean_rentals(df)
    model = train_model(df)
    if st.sidebar.button('Submit'):
        predicted = model.predict(input_df)
        rent_est = predicted[0][0]
        st.markdown(f'The rent estimate for this property is **${round(rent_est,0)}**')





main()