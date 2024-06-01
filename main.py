import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from PIL import Image
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
import os

# Load các mô hình và dữ liệu test
cluster_model = joblib.load('D:/Dương Diệu Linh_20070944/DSS/best_kmeans_model.pkl')
logistic_regression_model = pickle.load(open("D:/Dương Diệu Linh_20070944/DSS/logistic_regression_model.pkl", "rb"))
random_forest_model = pickle.load(open("D:/Dương Diệu Linh_20070944/DSS/random_forest_model.pkl", "rb"))
nn_model = load_model("D:/Dương Diệu Linh_20070944/DSS/neural_network_model.keras")
test_data = pd.read_csv("D:/Dương Diệu Linh_20070944/DSS/test.csv")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize session state for storing customer info
if 'customer_info' not in st.session_state:
    st.session_state.customer_info = []

# Initialize session state for customer ID
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = 1

# Function to save customer data
def save_customer_data(data, filename):
    customer_id = st.session_state.customer_id
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame()
    data['Customer_ID'] = customer_id
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(filename, index=False)
    # Increment customer_id for the next customer
    st.session_state.customer_id += 1

# Function to predict recommendation using different models
def predict_recommend(input_data, model_name):
    scaler = StandardScaler()
    scaler.fit(test_data.drop(columns=['Recommend']))
    input_data_scaled = scaler.transform(input_data)
    if model_name == "Logistic Regression":
        prediction = logistic_regression_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "Random Forest":
        prediction = random_forest_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "Neural Network":
        prediction = nn_model.predict(input_data_scaled)
    return prediction

# Function to encode Likert scale
def encode_likert(value):
    if value == "Dissatisfied":
        return 0
    elif value == "Neutral":
        return 1
    elif value == "Satisfied":
        return 2
    elif value == "Very dissatisfied":
        return 3
    elif value == "Very satisfied":
        return 4

# Main function
def main():
    if not st.session_state.logged_in:
        # Create two columns
        col1, col2 = st.columns([3, 2])  # Adjust the ratio as needed

        # Column for login form
        with col1:
            st.markdown("<h2 style='color:red;'>LOG IN</h2>", unsafe_allow_html=True)

            username = st.text_input("Username:")
            password = st.text_input("Password:", type="password")

            # Create the "Remember" checkbox and "Forgot Password?" link
            remember_col, forgot_password_col = st.columns([1, 2])
            with remember_col:
                remember = st.checkbox("Remember")
            with forgot_password_col:
                st.markdown("<a href='#'>Forgot Password?</a>", unsafe_allow_html=True)

            st.warning("username == Linh and password == Linh")

            # Log in button
            if st.button("Log in", key="login_button", help="Login with the provided credentials"):
                if username == "Linh" and password == "Linh":
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")

        # Column for image display
        with col2:
            image = Image.open("D:/Dương Diệu Linh_20070944/DSS/circle k 3.jpg")
            st.image(image, use_column_width=True)
    else:
        image = Image.open("D:/Dương Diệu Linh_20070944/DSS/circle K 2.jpg")
        st.image(image, use_column_width=True)
        st.title('Circle K Customer Behavior Cluster and Prediction WEBAPP')

        # Option menu for tabs
        with st.sidebar:
            selected = option_menu(
                "Main Menu",
                ["Basic Information", "Shopping Behavior", "Recommendation", "Segmentation", "Data Warehouse"],
                icons=["house", "shop", "feedback", "menu", "database"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "Basic Information":
            st.header("New Customer")
            customer_name = st.text_input("Customer Name", placeholder="Customer Name:")
            customer_gender = st.radio("Customer Gender", ["Male", "Female"])
            customer_age = st.selectbox("Age group", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
            customer_address = st.text_input("Address", placeholder="Address")
            customer_occupation = st.selectbox("Occupation", ["Student", "Go to work", "Retired"])
            purchase_frequency = st.multiselect("Know from", ["Relatives", "Friends", "Internet, TV", "Social Media (Facebook, Instagram, Tiktok, etc)", "Saw it by chance"])

            if st.button("Save"):
                basic_info = {
                    "Customer_Name": customer_name,
                    "Customer_Gender": customer_gender,
                    "Customer_Age": customer_age,
                    "Customer_Address": customer_address,
                    "Customer_Occupation": customer_occupation,
                    "Purchase_Frequency": ",".join(purchase_frequency)
                }
                save_customer_data(basic_info, "basic_information.csv")
                st.success("Basic Information Saved!")

        elif selected == "Shopping Behavior":
            st.header("Shopping Behavior")
            typically_purchase = st.multiselect("Typically Purchase", ["Drinks", "Grocery Items", "Snack Foods", "Tobacco Products", "Prepared Foods", "Health and Beauty", "Other"])
            feedback = st.text_area("Feedback")
            influences_visiting = st.multiselect("What influences your visiting?", ["Price", "Product variety", "Brand", "Atmosphere", "Customer service", "Promotions", "24/7"])

            if st.button("Save"):
                shopping_behavior = {
                    "Typically_Purchase": ",".join(typically_purchase),
                    "Feedback": feedback,
                    "Influences_Visiting": ",".join(influences_visiting)
                }
                save_customer_data(shopping_behavior, "shopping_behavior.csv")
                st.success("Shopping Behavior Saved!")

        elif selected == "Recommendation":
            st.subheader("Predict Recommendation")
            st.write("Please enter the following information for prediction:")

            # Information inputs for prediction
            good_feedback = st.selectbox("Good Feedback", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            variety_of_products = st.selectbox("Variety of Products", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            guarantees_quality_product = st.selectbox("Guarantees Quality Product", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            reasonable_price = st.selectbox("Reasonable Price", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            good_service = st.selectbox("Good Service", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            good_security = st.selectbox("Good Security", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            space = st.selectbox("Space", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            good_incentives = st.selectbox("Good Incentives", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            customer_care = st.selectbox("Customer Care", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            handles_complaints_well = st.selectbox("Handles Complaints Well", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            staff_attitude = st.selectbox("Staff Attitude", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])
            purchasing_experience_factors = st.selectbox("Purchasing Experience Factors", ["Dissatisfied", "Neutral", "Satisfied", "Very dissatisfied", "Very satisfied"])

            input_data = pd.DataFrame({
                "Good Feedback": [encode_likert(good_feedback)],
                "Variety of Products": [encode_likert(variety_of_products)],
                "Guarantees Quality Product": [encode_likert(guarantees_quality_product)],
                "Reasonable Price": [encode_likert(reasonable_price)],
                "Good Service": [encode_likert(good_service)],
                "Good Security": [encode_likert(good_security)],
                "Space": [encode_likert(space)],
                "Good Incentives": [encode_likert(good_incentives)],
                "Customer Care": [encode_likert(customer_care)],
                "Handles Complaints Well": [encode_likert(handles_complaints_well)],
                "Staff Attitude": [encode_likert(staff_attitude)],
                "Purchasing Experience Factors": [encode_likert(purchasing_experience_factors)]
            })

            model_name = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Neural Network"])

            if st.button("Predict"):
                prediction = predict_recommend(input_data, model_name)[0]
                recommendation_percentage = round(float(prediction) * 100, 2)
                if recommendation_percentage > 50:
                    recommendation_result = "Definitely Recommend"
                    st.success(f"Definitely Recommend Percentage: {recommendation_percentage}%")
                    st.success(f"Recommendation Result: {recommendation_result}")
                else:
                    recommendation_result = "NO Recommend"
                    st.error(f"Definitely Recommend Percentage: {recommendation_percentage}%")
                    st.error(f"Recommendation Result: {recommendation_result}")

                # Save prediction inputs and result to session state
                prediction_info = {
                    "Good Feedback": good_feedback,
                    "Variety of Products": variety_of_products,
                    "Guarantees Quality Product": guarantees_quality_product,
                    "Reasonable Price": reasonable_price,
                    "Good Service": good_service,
                    "Good Security": good_security,
                    "Space": space,
                    "Good Incentives": good_incentives,
                    "Customer Care": customer_care,
                    "Handles Complaints Well": handles_complaints_well,
                    "Staff Attitude": staff_attitude,
                    "Purchasing Experience Factors": purchasing_experience_factors,
                    "Model": model_name,
                    "Recommendation Percentage": recommendation_percentage,
                    "Recommendation Result": recommendation_result
                }
                save_customer_data(prediction_info, "recommend.csv")
                st.success("Feedback and Prediction Saved!")

        elif selected == "Segmentation":
            st.header("Cluster Customer")
            st.write("Please enter the following information for clustering:")
            st.warning("Time is the number of days from the last day that the service was used")

            # Information inputs for clustering
            time = st.number_input("Time", min_value=0)
            frequency = st.selectbox("Frequency", ["Daily", "Less than once a month", "Monthly", "Weekly"])
            estimated_monthly_spend = st.selectbox("Estimated Monthly Spend", ["Under 500,000 VND", "From 500,000 VND to 1,000,000 VND", "From 1,000,000 VND to 2,000,000 VND", "Over 2,000,000 VND"])

            def encode_frequency(value):
                if value == "Less than once a month":
                    return 1
                elif value == "Monthly":
                    return 2
                elif value == "Weekly":
                    return 3
                elif value == "Daily":
                    return 0

            def encode_estimated_monthly_spend(value):
                if value == "Under 500,000 VND":
                    return 2
                elif value == "From 500,000 VND to 1,000,000 VND":
                    return 1
                elif value == "From 1,000,000 VND to 2,000,000 VND":
                    return 0
                elif value == "Over 2,000,000 VND":
                    return 3

            cluster_names = {
                0: 'Champions (Power Shoppers)',
                1: 'Loyal Customers',
                2: 'At-risk Customers',
                3: 'Recent Customers'
            }

            input_data = pd.DataFrame({
                "R_Score": [time],
                "F_Score": [encode_frequency(frequency)],
                "M_Score": [encode_estimated_monthly_spend(estimated_monthly_spend)]
            })

            st.subheader("Cluster Information")
            st.warning("""
            - **Champions (Power Shoppers)**: Customers who bought most recently, most often, and are heavy spenders.
            - **Loyal Customers**: Customers who buy frequently but not necessarily recently or with high spending.
            - **At-risk Customers**: Customers who used to buy frequently but haven't bought recently.
            - **Recent Customers**: Customers who bought recently but not frequently or with high spending.
            """)

            if st.button("Cluster"):
                cluster = cluster_model.predict(input_data)[0]
                cluster_name = cluster_names.get(cluster, "Unknown")
                st.title('Cluster Result')
                st.success(f"**Cluster:** {cluster_name}")

                # Save cluster inputs and result to session state
                cluster_info = {
                    "Time": time,
                    "Frequency": frequency,
                    "Estimated_Monthly_Spend": estimated_monthly_spend,
                    "Cluster": cluster_name
                }
                save_customer_data(cluster_info, "segmentation.csv")
                st.success("Segmentation Saved!")

        elif selected == "Data Warehouse":
            st.subheader("Customer Data Warehouse")

            # Display all saved customer data
            st.write("Basic Information")
            if os.path.exists("basic_information.csv"):
                basic_info_df = pd.read_csv("basic_information.csv")
                st.dataframe(basic_info_df)

            st.write("Shopping Behavior")
            if os.path.exists("shopping_behavior.csv"):
                shopping_behavior_df = pd.read_csv("shopping_behavior.csv")
                st.dataframe(shopping_behavior_df)

            st.write("Recommend")
            if os.path.exists("recommend.csv"):
                recommend_df = pd.read_csv("recommend.csv")
                st.dataframe(recommend_df)

            st.write("Segmentation")
            if os.path.exists("segmentation.csv"):
                segmentation_df = pd.read_csv("segmentation.csv")
                st.dataframe(segmentation_df)

            # Option to download the data as CSV
            if os.path.exists("basic_information.csv"):
                st.download_button(
                    label="Download Basic Information as CSV",
                    data=basic_info_df.to_csv(index=False).encode('utf-8'),
                    file_name='basic_information.csv',
                    mime='text/csv'
                )

            if os.path.exists("shopping_behavior.csv"):
                st.download_button(
                    label="Download Shopping Behavior as CSV",
                    data=shopping_behavior_df.to_csv(index=False).encode('utf-8'),
                    file_name='shopping_behavior.csv',
                    mime='text/csv'
                )

            if os.path.exists("recommend.csv"):
                st.download_button(
                    label="Download Recommend as CSV",
                    data=recommend_df.to_csv(index=False).encode('utf-8'),
                    file_name='recommend.csv',
                    mime='text/csv'
                )

            if os.path.exists("segmentation.csv"):
                st.download_button(
                    label="Download Segmentation as CSV",
                    data=segmentation_df.to_csv(index=False).encode('utf-8'),
                    file_name='segmentation.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()