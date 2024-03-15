import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
import joblib

model_path = 'Best_model.joblib'
loaded_model = joblib.load(model_path)


# Preprocess input function
def preprocess_input(input_data):
    age = input_data['age']
    bmi = input_data.get('bmi', None)
    height = input_data.get('height', None)
    weight = input_data.get('weight', None)
    children = input_data['children']

    # Convert height to meters based on the selected unit
    height_unit = input_data.get('height_unit', 'meters')
    if height is not None and height_unit != 'meters':
        if height_unit == 'centimeters':
            height /= 100
        elif height_unit == 'feet':
            height *= 0.3048  # 1 foot = 0.3048 meters

    # Calculate BMI if height and weight are provided and height is not zero
    if height is not None and height != 0 and weight is not None:
        bmi = weight / (height ** 2)

    # Convert sex to binary representation
    sex_0 = 1 if input_data['sex'] == 'female' else 0
    sex_1 = 1 - sex_0

    # Convert smoker to binary representation
    smoker_0 = 1 if input_data['smoker'] == 'no' else 0
    smoker_1 = 1 - smoker_0

    # Map region name to numerical representation
    region_mapping = {'southeast': 1, 'southwest': 2, 'northwest': 3, 'northeast': 4}
    region = region_mapping.get(input_data['region'], 0)

    # Create binary representations for each region
    region_1 = 1 if region == 1 else 0
    region_2 = 1 if region == 2 else 0
    region_3 = 1 if region == 3 else 0
    region_4 = 1 if region == 4 else 0

    # Create the formatted input list with 11 features
    formatted_input = [age, bmi, children, sex_0, sex_1, region_1, region_2, region_3, region_4, smoker_0, smoker_1]

    return formatted_input


# Input page
def input_page():
    st.title('Health Insurance Price Prediction')
    st.write('Please fill in the following details:')
    age = st.number_input('Age', min_value=0, step=1)
    sex = st.radio('Sex', ('male', 'female'))

    # Side-by-side input for height unit and height
    col1, col2 = st.columns(2)
    with col1:
        height_unit = st.selectbox('Height Unit', ('meters', 'centimeters', 'feet'))
    with col2:
        height = st.number_input('Height', min_value=0.0, step=0.01)
    weight = st.number_input('Weight (in kg)', min_value=0.0, step=0.1)

    # Calculate BMI immediately after entering height and weight if height is not zero
    bmi = None
    if height is not None and height != 0.0 and weight is not None:
        # Convert height based on selected height unit
        if height_unit != 'meters':
            if height_unit == 'centimeters':
                height /= 100
            elif height_unit == 'feet':
                height *= 0.3048  # 1 foot = 0.3048 meters

        # Calculate BMI
        bmi = weight / (height ** 2)
        st.write(f'BMI: {bmi:.2f}')

    children = st.number_input('Number of Children', min_value=0, step=1)
    smoker = st.selectbox('Smoker', ('yes', 'no'))
    region = st.selectbox('Region', ('southeast', 'southwest', 'northwest', 'northeast'))

    if st.button('Predict'):
        input_data = {'age': age, 'sex': sex, 'height': height, 'weight': weight, 'children': children,
                      'smoker': smoker, 'region': region, 'bmi': bmi, 'height_unit': height_unit}
        processed_input = preprocess_input(input_data)
        charges = loaded_model.predict([processed_input])[0]
        st.write('## Estimated Claim Amount')
        st.write(f'Estimated Claim Amount: {charges:.2f}', unsafe_allow_html=True)
        st.write('The following value is estimated based on historical data and predictive modeling techniques and may not represent the exact amount.')


# Main function
def main():
    input_page()


if __name__ == '__main__':
    main()
