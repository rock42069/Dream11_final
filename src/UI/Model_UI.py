import streamlit as st
import os
import pandas as pd
from datetime import datetime
from data_processing.data_download import download_and_preprocess
from data_processing.feature_generation import main_feature_generation
from model.train_model import main_train_and_save
from model.predict_model import main_generate_predictions
from model.final_runner import main_generate_csv

current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the main logic for the Streamlit app
def main():
    st.set_page_config(page_title="Cricket Match Prediction System", layout="wide")

    st.title("Model UI for Dream11 Score Prediction System")
    st.markdown("""
        We aim to design a state-of-the-art model that will predict the best possible 11 players taking all possible factors into account. 
        Our model will also be explainable so that all cricket fans can make sense of our predictions.
        Along with these, we aim to build an attractive and intuitive user interface that will be easy to use for all users.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup", "Prediction Workflow", "Model Insights"])

    if "setup_done" not in st.session_state:
        st.session_state.setup_done = False

    # Setup Page
    if page == "Setup":
        st.header("⚙️ Setup")
        st.write("Download and preprocess data, and generate features for training.")

        if st.button("Run Setup"):
            # Add logic for setup here (e.g., data download, preprocessing)
            with st.spinner("Downloading and preprocessing data..."):
                download_and_preprocess()
            with st.spinner("Generating features..."):
                main_feature_generation()
            st.session_state.setup_done = True
            st.success("Setup completed successfully! ✅")

    # Prediction Workflow Page
    elif page == "Prediction Workflow":
        st.header("📊 Prediction Workflow")

        if not st.session_state.setup_done:
            st.warning("Please complete the setup first.")
        else:
            st.subheader("📅 Input Date Ranges")
            st.info("Enter the training and testing date ranges below:")

            col1, col2 = st.columns(2)
            train_start_date = col1.date_input("Training Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
            train_end_date = col2.date_input("Training End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

            col3, col4 = st.columns(2)
            test_start_date = col3.date_input("Testing Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
            test_end_date = col4.date_input("Testing End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

            if train_start_date >= train_end_date:
                st.error("Training Start Date must be before Training End Date.")
            elif test_start_date >= test_end_date:
                st.error("Testing Start Date must be before Testing End Date.")
            else:
                if st.button("Run Prediction Workflow"):
                    # Add logic for training and prediction here
                    if not st.session_state.get('model_trained', False):
                        with st.spinner("Training models..."):
                            main_train_and_save(str(train_start_date), str(train_end_date))
                        st.session_state.model_trained = True

                    with st.spinner("Generating predictions..."):
                        main_generate_predictions(str(train_start_date), str(train_end_date), str(test_start_date), str(test_end_date))

                    with st.spinner("Processing matches..."):
                        main_generate_csv()
                    st.session_state.train_start_date = str(train_start_date)
                    st.session_state.train_end_date = str(train_end_date)

                    # Mock final output for demonstration
                    final_output = pd.DataFrame({"Player": ["Player 1", "Player 2"], "Score": [100, 90]})
                    final_output_path = os.path.join(current_dir, "data", "processed", "final_output.csv")
                    # Uncomment below if file handling logic is implemented
                    final_output = pd.read_csv(final_output_path)
                    st.success("Prediction workflow completed! ✅")

                    # Display and Download Final Output
                    st.subheader("📜 Final Output")
                    st.dataframe(final_output)
                    st.download_button(
                        label="Download Final Output CSV",
                        data=final_output.to_csv(index=False),
                        file_name="final_output.csv",
                        mime="text/csv",
                    )

    # Model Insights Page
    elif page == "Model Insights":
        st.header("📈 Model Insights with SHAP")

        if not st.session_state.setup_done:
            st.warning("Please complete the setup first.")
        else:

            if "train_start_date" in st.session_state and "train_end_date" in st.session_state:
                train_start_date = st.session_state.train_start_date
                train_end_date = st.session_state.train_end_date

            model_type = st.selectbox("Select Model Type", ["odi", "test", "t20"])
            shap_plot_path = os.path.join(
                current_dir, "data", "raw", "additional_data",
                f"{model_type}_{train_start_date}_{train_end_date}_shap_summary_plot.png"
            )

            if os.path.exists(shap_plot_path):
                st.success(f"{model_type.upper()} model SHAP summary plot loaded successfully!")
                st.image(shap_plot_path, caption=f"SHAP Summary Plot for {model_type.upper()} Model")
            else:
                st.warning(f"SHAP summary plot for {model_type.upper()} model not found.")

