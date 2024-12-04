import streamlit as st
import os
import pandas as pd
from datetime import datetime
from src.data_processing.data_download import download_and_preprocess
from src.data_processing.feature_generation import main_feature_generation
from src.model.train_model import main_train_and_save
from src.model.predict_model import main_generate_predictions
from src.model.final_runner import main_process_matches
import os
import pandas as pd
import streamlit as st
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from joblib import load

current_dir = os.path.dirname(__file__)

import os
import pandas as pd
import streamlit as st
import shap
from datetime import datetime
from joblib import load
import matplotlib.pyplot as plt

# current_dir = os.path.dirname(__file__)

# # Load SHAP explainer
# def load_explainer(model_path):
#     model = load(model_path)
#     explainer = shap.Explainer(model)
#     return explainer

# # Define the main logic for the Streamlit app
# def main():
#     st.set_page_config(page_title="Cricket Match Prediction System", layout="wide")

#     st.title("Model UI for Dream11 Score Prediction System")
#     st.markdown("""
#             We aim to design a state-of-the-art model that will predict the best possible 11 players taking all possible factors into account. 
#             Our model will also be explainable so that all cricket fans can make sense of our predictions.
#             Along with these, we aim to build an attractive and intuitive user interface that will be easy to use for all users.
#     """)

#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Setup", "Train Model", "Run Predictions", "Model Insights"])

#     if "setup_done" not in st.session_state:
#         st.session_state.setup_done = False
#     if "model_trained" not in st.session_state:
#         st.session_state.model_trained = False
#     if "train_start_date" not in st.session_state:
#         st.session_state.train_start_date = None
#     if "train_end_date" not in st.session_state:
#         st.session_state.train_end_date = None

#     # Setup Page
#     if page == "Setup":
#         st.header("⚙️ Setup")
#         st.write("Download and preprocess data, and generate features for training.")

#         if st.button("Run Setup"):
#             with st.spinner("Downloading and preprocessing data..."):
#                 download_and_preprocess()
#             with st.spinner("Generating features..."):
#                 main_feature_generation()
#             st.session_state.setup_done = True
#             st.success("Setup completed successfully! ✅")

#     # Train Model Page
#     elif page == "Train Model":
#         st.header("🏋️ Train Model")

#         if not st.session_state.setup_done:
#             st.warning("Please complete the setup first.")
#         else:
#             st.subheader("📅 Input Training Date Ranges")
#             col1, col2 = st.columns(2)
#             st.session_state.train_start_date = col1.date_input("Training Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
#             st.session_state.train_end_date = col2.date_input("Training End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

#             if st.session_state.train_start_date >= st.session_state.train_end_date:
#                 st.error("Training Start Date must be before Training End Date.")

#             if st.button("Train Model"):
#                 with st.spinner("Training models..."):
#                     main_train_and_save(str(st.session_state.train_start_date), str(st.session_state.train_end_date))
#                 st.session_state.model_trained = True
#                 st.success("Model trained and saved successfully! ✅")

#     # Run Predictions Page
#     elif page == "Run Predictions":
#         st.header("📊 Run Predictions")

#         if not st.session_state.model_trained:
#             st.warning("Please train the model first.")
#         else:
#             st.subheader("📅 Input Testing Date Ranges")
#             col1, col2 = st.columns(2)
#             test_start_date = col1.date_input("Testing Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
#             test_end_date = col2.date_input("Testing End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

#             if test_start_date >= test_end_date:
#                 st.error("Testing Start Date must be before Testing End Date.")

#             if st.button("Run Predictions"):
#                 with st.spinner("Generating predictions..."):
#                     main_generate_predictions(str(st.session_state.train_start_date), str(st.session_state.train_end_date), str(test_start_date), str(test_end_date))

#                 # Load and display the final output
#                 final_output_path = os.path.join(current_dir, "src", "data", "processed", "final_output.csv")
#                 if os.path.exists(final_output_path):
#                     final_output = pd.read_csv(final_output_path)
#                     st.success("Predictions generated successfully! ✅")
#                     st.subheader("📜 Final Output")
#                     st.dataframe(final_output)
#                     st.download_button(
#                         label="Download Final Output CSV",
#                         data=final_output.to_csv(index=False),
#                         file_name="final_output.csv",
#                         mime="text/csv",
#                     )

#     # Model Insights with SHAP
#     elif page == "Model Insights":
#         st.header("📈 Model Insights")
#         st.write("Explore feature importance with SHAP values for better interpretability.")

#         model_path = os.path.join(current_dir, "src", "model_artifacts", "Model_UI_2000_01_01-2022_01_01.pkl")
#         # explainer = load_explainer(model_path)

#         sample_data_path = os.path.join(current_dir, "src", "data", "processed", "final_training_file_ODI.csv")
#         sample_data = pd.read_csv(sample_data_path)

#         st.write("### Sample Data Used for Predictions")
#         st.dataframe(sample_data)

#         # if st.button("Generate SHAP Summary Plot"):
#         #     with st.spinner("Generating SHAP summary plot..."):
#         #         shap_values = explainer(sample_data)
#         #         fig, ax = plt.subplots(figsize=(10, 6))
#         #         shap.summary_plot(shap_values, sample_data, show=False)
#         #         st.pyplot(fig)

#         # if st.button("Generate SHAP Dependence Plot"):
#         #     feature = st.selectbox("Select Feature for Dependence Plot", sample_data.columns)
#         #     with st.spinner("Generating SHAP dependence plot..."):
#         #         fig, ax = plt.subplots(figsize=(8, 5))
#         #         shap.dependence_plot(feature, shap_values.values, sample_data, ax=ax)
#         #         st.pyplot(fig)

# if __name__ == "__main__":
#     main()


# Load SHAP explainer
def load_explainer(model_path):
    model = load(model_path)
    explainer = shap.Explainer(model)
    return explainer

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
            with st.spinner("Downloading and preprocessing data..."):
                download_and_preprocess()
            with st.spinner("Generating features..."):
                main_feature_generation()
            st.session_state.setup_done = True
            st.success("Setup completed successfully! ✅")

    # Prediction Workflows
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
            if test_start_date >= test_end_date:
                st.error("Testing Start Date must be before Testing End Date.")

            if st.button("Run Prediction Workflow"):
                with st.spinner("Training models..."):
                    main_train_and_save(str(train_start_date), str(train_end_date))

                with st.spinner("Generating predictions..."):
                    main_generate_predictions(str(train_start_date), str(train_end_date), str(test_start_date), str(test_end_date))

                with st.spinner("Processing matches..."):
                    main_process_matches()

                # Merge final outputs
                final_output_odi = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_odi.csv"))
                final_output_t20 = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_t20.csv"))
                final_output_test = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_test.csv"))

                final_output = pd.concat([final_output_odi, final_output_t20, final_output_test], ignore_index=True)
                final_output_path = os.path.join(current_dir, "src", "data", "processed", "final_output.csv")
                final_output.to_csv(final_output_path, index=False)

                # Clean up intermediate files
                # for file in ["final_output_odi.csv", "final_output_t20.csv", "final_output_test.csv", "predictions_odi.csv", "predictions_t20.csv", "predictions_test.csv"]:
                #     os.remove(os.path.join(current_dir, "src", "data", "processed", file))

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

    # Model Insights with SHAP
    elif page == "Model Insights":
        st.header("📈 Model Insights")
        st.write("Explore feature importance with SHAP values for better interpretability.")

        model_path = os.path.join(current_dir, "src", "model_artifacts", "Model_UI_2000_01_01-2022_01_01.pkl")
        # explainer = load_explainer(model_path)

        sample_data_path = os.path.join(current_dir, "src", "data", "processed", "final_training_file_ODI.csv")
        sample_data = pd.read_csv(sample_data_path)

        st.write("### Sample Data Used for Predictions")
        st.dataframe(sample_data)

        # if st.button("Generate SHAP Summary Plot"):
        #     with st.spinner("Generating SHAP summary plot..."):
        #         shap_values = explainer(sample_data)
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         shap.summary_plot(shap_values, sample_data, show=False)
        #         st.pyplot(fig)

        # if st.button("Generate SHAP Dependence Plot"):
        #     feature = st.selectbox("Select Feature for Dependence Plot", sample_data.columns)
        #     with st.spinner("Generating SHAP dependence plot..."):
        #         fig, ax = plt.subplots(figsize=(8, 5))
        #         shap.dependence_plot(feature, shap_values.values, sample_data, ax=ax)
        #         st.pyplot(fig)

if __name__ == "__main__":
    main()

# current_dir = os.path.dirname(__file__)

# # Define the main logic for the Streamlit app
# def main():
#     st.title("Cricket Match Prediction System")

#     # Setup Step
#     # Setup Page
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Setup", "Prediction Workflow"])

#     if "setup_done" not in st.session_state:
#         st.session_state.setup_done = False

#     if page == "Setup":
#         st.header("Setup")
#         if st.button("Run Setup"):
#             with st.spinner("Downloading and preprocessing data..."):
#                 download_and_preprocess()
#             with st.spinner("Generating features..."):
#                 main_feature_generation()
#             st.session_state.setup_done = True
#             st.success("Setup completed successfully!")

#     elif page == "Prediction Workflow":
#         if not st.session_state.setup_done:
#             st.warning("Please complete the setup first.")
#         else:
#             st.header("Input Date Ranges")
#             st.info("Enter the training and testing date ranges below:")

#             col1, col2 = st.columns(2)
#             train_start_date = col1.date_input("Training Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
#             train_end_date = col2.date_input("Training End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

#             col3, col4 = st.columns(2)
#             test_start_date = col3.date_input("Testing Start Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))
#             test_end_date = col4.date_input("Testing End Date", datetime(2000, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 12, 31))

#             if train_start_date >= train_end_date:
#                 st.error("Training Start Date must be before Training End Date.")
#             if test_start_date >= test_end_date:
#                 st.error("Testing Start Date must be before Testing End Date.")

#             if st.button("Run Prediction Workflow"):
#                 with st.spinner("Training models..."):
#                     main_train_and_save(str(train_start_date), str(train_end_date))

#                 with st.spinner("Generating predictions..."):
#                     main_generate_predictions(str(train_start_date), str(train_end_date), str(test_start_date), str(test_end_date))

#                 with st.spinner("Processing matches..."):
#                     main_process_matches()

#                 # Merge final outputs
#                 final_output_odi = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_odi.csv"))
#                 final_output_t20 = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_t20.csv"))
#                 final_output_test = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_test.csv"))

#                 final_output = pd.concat([final_output_odi, final_output_t20, final_output_test], ignore_index=True)
#                 final_output_path = os.path.join(current_dir, "src", "data", "processed", "final_output.csv")
#                 final_output.to_csv(final_output_path, index=False)

#                 # Clean up intermediate files
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_odi.csv"))
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_t20.csv"))
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_test.csv"))
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_odi.csv"))
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_t20.csv"))
#                 os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_test.csv"))

#                 st.success("Prediction workflow completed!")

#                 # Display and Download Final Output
#                 st.header("Final Output")
#                 st.dataframe(final_output)
#                 st.download_button(
#                     label="Download Final Output CSV",
#                     data=final_output.to_csv(index=False),
#                     file_name="final_output.csv",
#                     mime="text/csv",
#                 )

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()


# def main(train_start_date, train_end_date, test_start_date, test_end_date): 

#     download_and_preprocess()
#     main_feature_generation()

#     print("Training and saving the model")
#     train_and_save_model_odi(train_start_date, train_end_date)
#     train_and_save_model_t20(train_start_date, train_end_date)
#     train_and_save_model_test(train_start_date, train_end_date)

#     print("Generating predictions")
#     generate_predictions_t20(train_start_date, train_end_date, test_start_date, test_end_date)
#     generate_predictions_odi(train_start_date, train_end_date, test_start_date, test_end_date)
#     generate_predictions_test(train_start_date, train_end_date, test_start_date, test_end_date)

#     print("Processing matches and generating final output")
#     process_matches("test")
#     process_matches("t20")
#     process_matches("odi")

#     # Merge the final output files
#     final_output_odi = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_odi.csv"))
#     final_output_t20 = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_t20.csv"))
#     final_output_test = pd.read_csv(os.path.join(current_dir, "src", "data", "processed", "final_output_test.csv"))

#     final_output = pd.concat([final_output_odi, final_output_t20, final_output_test], ignore_index=True)
#     final_output.to_csv(os.path.join(current_dir, "src", "data", "processed", "final_output.csv"), index=False)

#     os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_odi.csv"))
#     os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_t20.csv"))
#     os.remove(os.path.join(current_dir, "src", "data", "processed", "final_output_test.csv"))
#     os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_odi.csv"))
#     os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_t20.csv"))
#     os.remove(os.path.join(current_dir, "src", "data", "processed", "predictions_test.csv"))

#     return


# main("2020-01-01", "2023-01-01", "2024-01-01", "2024-10-01")

# # State Management
# if "setup_done" not in st.session_state:
#     st.session_state.setup_done = False

# # Streamlit UI
# st.title("Pipeline Automation for Cricket Match Predictions")

# # Initial Setup Section
# st.markdown("### Step 1: Initial Setup")
# if not st.session_state.setup_done:
#     st.write("This step includes downloading data and generating features. Once completed, you can proceed to the next steps.")

#     if st.button("Run Initial Setup"):
#         try:
#             st.write("Downloading and preprocessing data...")
#             download_and_preprocess()
#             st.write("Generating features...")
#             generate_features_test()
#             st.session_state.setup_done = True
#             st.success("Initial setup completed successfully!")
#         except Exception as e:
#             st.error(f"An error occurred during setup: {e}")
# else:
#     st.success("Initial setup already completed.")

# # Repeatable Steps Section
# st.markdown("### Step 2: Run Pipeline")
# if st.session_state.setup_done:
#     with st.form("repeatable_steps_form"):
#         train_start_date = st.date_input("Training Start Date")
#         train_end_date = st.date_input("Training End Date")
#         test_start_date = st.date_input("Testing Start Date")
#         test_end_date = st.date_input("Testing End Date")
#         submitted = st.form_submit_button("Run Pipeline")

#     if submitted:
#         try:
#             # Define steps and estimated times (in seconds)
#             steps = [
#                 (
#                     "Training and saving the model",
#                     lambda: train_and_save_model(train_start_date.strftime("%Y-%m-%d"), train_end_date.strftime("%Y-%m-%d")),
#                     20,
#                 ),
#                 (
#                     "Generating predictions",
#                     lambda: generate_predictions(
#                         train_start_date.strftime("%Y-%m-%d"),
#                         train_end_date.strftime("%Y-%m-%d"),
#                         test_start_date.strftime("%Y-%m-%d"),
#                         test_end_date.strftime("%Y-%m-%d"),
#                     ),
#                     15,
#                 ),
#                 ("Processing matches and generating final output", lambda: process_matches(), 5),
#             ]

#             # Initialize progress bar and timer
#             progress = st.progress(0)
#             total_time = sum(step[2] for step in steps)
#             start_time = datetime.now()

#             # Execute steps with progress updates
#             for idx, (description, func, estimated_time) in enumerate(steps):
#                 st.write(f"Step {idx + 1}/{len(steps)}: {description}")
#                 func()
#                 elapsed = (datetime.now() - start_time).total_seconds()
#                 progress.progress(int(((idx + 1) / len(steps)) * 100))
#                 st.write(
#                     f"Estimated time remaining: {str(timedelta(seconds=total_time - elapsed)).split('.')[0]}"
#                 )
#                 time.sleep(1)  # Simulate processing delay for UX (optional)

#             st.success("Pipeline completed successfully!")
#             st.write(f"Download the final output below:")

#             # File download button
#             output_path = os.path.abspath(
#                 os.path.join(os.path.dirname(__file__), "src", "data", "processed", "final_output.csv")
#             )
#             # output_path = "/Users/trijalsrivastava/Code/Dream11_final/src/data/processed/final_output.csv"
#             with open(output_path, "rb") as file:
#                 st.download_button(
#                     label="Download Final Output CSV",
#                     data=file,
#                     file_name="final_output.csv",
#                     mime="text/csv",
#                 )

#         except Exception as e:
#             st.error(f"An error occurred: {e}")
# else:
#     st.warning("Please complete the initial setup first.")




