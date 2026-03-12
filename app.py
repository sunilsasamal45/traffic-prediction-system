import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("traffic_model.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))

st.title("🚦 Smart Traffic Prediction System")

car = st.number_input("Car Count",0)
bike = st.number_input("Bike Count",0)
bus = st.number_input("Bus Count",0)
truck = st.number_input("Truck Count",0)

hour = st.slider("Hour",0,23)
day = st.slider("Day of Week",0,6)

if st.button("Predict Traffic"):

    total = car + bike + bus + truck

    data = pd.DataFrame(
        [[car,bike,bus,truck,total,hour,day]],
        columns=["CarCount","BikeCount","BusCount","TruckCount","Total","Hour","Day of the week"]
    )

    prediction = model.predict(data)

    traffic = le.inverse_transform(prediction)


    st.success("Predicted Traffic: " + traffic[0])
