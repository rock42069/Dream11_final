import streamlit as st
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from src.data_processing.data_download import download_and_preprocess
from src.data_processing.feature_generation import generate_features
from src.model.train_model import train_and_save_model
from src.model.predict_model import generate_predictions
from src.model.final_runner import process_matches

# State Management
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False

# Streamlit UI
st.title("Pipeline Automation for Cricket Match Predictions")

# Initial Setup Section
st.markdown("### Step 1: Initial Setup")
if not st.session_state.setup_done:
    st.write("This step includes downloading data and generating features. Once completed, you can proceed to the next steps.")

    if st.button("Run Initial Setup"):
        try:
            st.write("Downloading and preprocessing data...")
            # download_and_preprocess()
            st.write("Generating features...")
            generate_features()
            st.session_state.setup_done = True
            st.success("Initial setup completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during setup: {e}")
else:
    st.success("Initial setup already completed.")

# Repeatable Steps Section
st.markdown("### Step 2: Run Pipeline")
if st.session_state.setup_done:
    with st.form("repeatable_steps_form"):
        train_start_date = st.date_input("Training Start Date")
        train_end_date = st.date_input("Training End Date")
        test_start_date = st.date_input("Testing Start Date")
        test_end_date = st.date_input("Testing End Date")
        submitted = st.form_submit_button("Run Pipeline")

    if submitted:
        try:
            # Define steps and estimated times (in seconds)
            steps = [
                # (
                #     "Training and saving the model",
                #     lambda: train_and_save_model(train_start_date.strftime("%Y-%m-%d"), train_end_date.strftime("%Y-%m-%d")),
                #     20,
                # ),
                (
                    "Generating predictions",
                    lambda: generate_predictions(
                        train_start_date.strftime("%Y-%m-%d"),
                        train_end_date.strftime("%Y-%m-%d"),
                        test_start_date.strftime("%Y-%m-%d"),
                        test_end_date.strftime("%Y-%m-%d"),
                    ),
                    15,
                ),
                ("Processing matches and generating final output", lambda: process_matches(), 5),
            ]

            # Initialize progress bar and timer
            progress = st.progress(0)
            total_time = sum(step[2] for step in steps)
            start_time = datetime.now()

            # Execute steps with progress updates
            for idx, (description, func, estimated_time) in enumerate(steps):
                st.write(f"Step {idx + 1}/{len(steps)}: {description}")
                func()
                elapsed = (datetime.now() - start_time).total_seconds()
                progress.progress(int(((idx + 1) / len(steps)) * 100))
                st.write(
                    f"Estimated time remaining: {str(timedelta(seconds=total_time - elapsed)).split('.')[0]}"
                )
                time.sleep(1)  # Simulate processing delay for UX (optional)

            st.success("Pipeline completed successfully!")
            st.write(f"Download the final output below:")

            # File download button
            output_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "src", "data", "processed", "final_output.csv")
            )
            # output_path = "/Users/trijalsrivastava/Code/Dream11_final/src/data/processed/final_output.csv"
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Final Output CSV",
                    data=file,
                    file_name="final_output.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Please complete the initial setup first.")
