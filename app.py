import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the XGBRegressor model
filename = 'diamond_price_prediction_tree_model.pkl'
model = pickle.load(open(filename, 'rb'))

st.title('Diamond Price prediction')

column_values = {
    'CUT': ['Round', 'Pear', 'Oval', 'Marquise', 'Princess', 'Emerald', 'Heart', 'Cushion', 'Radiant', 'Cushion Modified', 'Asscher'],
    'COLOR': ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
    'CLARITY': ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
    'CARAT WEIGHT': [],
    'CUT QUALITY': ['Fair', 'Good', 'Very Good', 'Excellent', 'Ideal'],
    'SYMNETRY': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
    'POLISH': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
    'DEPTH PERCENT': [],
    'TABLE PERCENT': [],
    'MEAS LENGTH': [],
    'MEAS WIDTH': [],
    'MEAS DEPTH': []
}

descriptions = {
    'CARAT WEIGHT': 'Carat weight refers to the size of the diamond, with 1 carat being equivalent to 200 milligrams.',
    'CUT': 'The cut refers to the shape and style of the diamond, which affects its brilliance and sparkle.',
    'COLOR': 'Diamond color is graded on a scale from D (colorless) to Z (light yellow or brown). D is the most valuable and rarest color.',
    'CLARITY': 'Clarity measures the presence of internal and external flaws (inclusions and blemishes). It is graded on a scale from I3 (most visible) to IF (internally flawless).',
    'CUT QUALITY': 'Cut quality describes how well a diamond has been cut, determining its ability to reflect light.',
    'SYMNETRY': 'Symmetry refers to the alignment and balance of a diamond\'s facets.',
    'POLISH': 'Polish is a measure of the smoothness and quality of the diamond\'s surface.',
    'DEPTH PERCENT': 'Depth percent is the ratio of the depth of the diamond to its average width.',
    'TABLE PERCENT': 'Table percent is the ratio of the width of the diamond\'s top facet (table) to its average width.',
    'MEAS LENGTH': 'Measurement of the diamond\'s length in millimeters.',
    'MEAS WIDTH': 'Measurement of the diamond\'s width in millimeters.',
    'MEAS DEPTH': 'Measurement of the diamond\'s depth in millimeters.'}

inputs = {}

for feature, values in column_values.items():
    if feature in ['CARAT WEIGHT', 'DEPTH PERCENT', 'TABLE PERCENT', 'MEAS LENGTH', 'MEAS WIDTH', 'MEAS DEPTH']:
        # Input for numerical features
        inputs[feature] = st.number_input(f'Enter {feature}', value=0.0)
    else:
        # Input for categorical features
        inputs[feature] = st.selectbox(f'Select {feature}', values)

    # Display description and image for the feature
    st.write(f'**Description of {feature}:**')
    st.write(descriptions.get(feature, ''))
    if feature in['CUT','CARAT','COLOR','CLARITY','DEPTH PERCENT']:
        st.image(f'{feature.lower()}.png')

# Perform one-hot encoding for categorical features
enc = OneHotEncoder(handle_unknown='ignore')
encoded_inputs = []
for feature, values in column_values.items():
    if feature not in ['CARAT WEIGHT', 'DEPTH PERCENT', 'TABLE PERCENT', 'MEAS LENGTH', 'MEAS WIDTH', 'MEAS DEPTH']:
        encoded_feature = enc.fit_transform(np.array(inputs[feature]).reshape(-1, 1)).toarray()
        encoded_inputs.append(encoded_feature)

# Create a NumPy array of the user input
numerical_inputs = np.array([inputs[feature] for feature in ['CARAT WEIGHT', 'DEPTH PERCENT', 'TABLE PERCENT', 'MEAS LENGTH', 'MEAS WIDTH', 'MEAS DEPTH']]).reshape(1, -1)
encoded_inputs = np.concatenate(encoded_inputs, axis=1)
input_array = np.concatenate([numerical_inputs, encoded_inputs], axis=1)

if st.button('Submit'):
    prediction = model.predict(input_array)

    # Display the prediction
    st.write('The predicted price of the diamond is: $' + str(prediction[0]))
