import pandas as pd
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def drop_zero_dominant_rows(df, threshold=0.9):
    # Calculate the number of zeros in each row
    zero_count = (df == 0).sum(axis=1)

    # Calculate the threshold number of zeros allowed per row (2/3 of the columns)
    threshold_zeros = int(threshold * df.shape[1])  # Number of zeros allowed

    # Drop rows where the number of zeros exceeds the threshold
    df_cleaned = df[zero_count <= threshold_zeros]

    return df_cleaned

def train_models(X_train, y_train):
    base_models = [
        ("linear_regression", LinearRegression()),
        ('catboost', CatBoostRegressor(verbose=0, random_state=42)),
        ("mlp", MLPRegressor(random_state=42, max_iter=1000))
    ]

    meta_model = LGBMRegressor(random_state = 42)

    # Reshape X_train and y_train to ensure proper dimensions for each model
    if len(X_train.shape) == 1:  # If X_train is a 1D array, convert it to 2D
        X_train = X_train.reshape(-1, 1)

    if len(y_train.shape) == 1:  # If y_train is a 1D array, no reshaping needed
        pass
    elif len(y_train.shape) == 2:  # If y_train is 2D, flatten it
        y_train = y_train.ravel()
    # Fit the model
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacked_model.fit(X_train, y_train)

    return stacked_model

def split(df, date):
    """
    Split the dataset into training and testing sets based on a given date.

    Parameters:
    - df (DataFrame): The DataFrame to be split, with a 'start_date' column.
    - date (str): The date to split the data on. Rows with 'start_date' less than or equal to this date 
                  will be in the training set, and rows with a 'start_date' greater than this date 
                  will be in the test set.

    Returns:
    - train (DataFrame): The training set, containing rows with 'start_date' <= date.
    - test (DataFrame): The testing set, containing rows with 'start_date' > date.
    """
    # Convert the 'start_date' column to datetime if it's not already
    df['start_date'] = pd.to_datetime(df['start_date'])

    # Split the data based on the given date
    train = df[df['start_date'] <= pd.to_datetime(date)]
    test = df[df['start_date'] > pd.to_datetime(date)]

    return train, test

def RFE_selector_regression(X_train_scaled, y_train, X_test_scaled, n_features=57):
    """
    Perform Recursive Feature Elimination (RFE) for regression tasks
    and ensure consistent feature selection between train and test sets.

    Parameters:
    - X_train_scaled: Scaled training feature data (DataFrame).
    - y_train: Target values for training.
    - X_test_scaled: Scaled test feature data (DataFrame).
    - n_features: Number of features to select (default=5).

    Returns:
    - X_train_selected: Transformed training data with selected features (DataFrame).
    - X_test_selected: Transformed test data with selected features (DataFrame).
    - selected_columns: List of selected feature names.
    """
    # Initialize Linear Regression model
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features)

    # Fit the selector on training data
    selector.fit(X_train_scaled, y_train)

    # Extract selected feature names
    selected_columns = X_train_scaled.columns[selector.support_]

    # Transform train and test data with consistent feature order
    X_train_selected = X_train_scaled[selected_columns]
    X_test_selected = X_test_scaled[selected_columns]

    return X_train_selected, X_test_selected, selected_columns


def predict_scores(trained_model, X_train, X_test):
    # Ensure columns of X_test align with X_train columns
    X_test = X_test[X_train.columns]

    test_data = pd.DataFrame()

    # Predict scores using the trained stacking model
    pred_scores = trained_model.predict(X_test)  # Predict the scores

    # Store the predicted scores in the DataFrame
    test_data['stacked_model_predicted_score'] = pred_scores

    return test_data

def predictions_per_match(trained_models, X_train, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores(trained_models, X_train, X_test)

    # Reset indices of test and predictions for alignment
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    # Assign match_id and fantasy_score_total from test to predictions DataFrame
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')
    predictions['match_type'] = test_reset.get('match_type')


    return predictions, test_reset


def avg_percentage_error_per_model(predictions):
    percentage_error_per_model = {model_name: 0 for model_name in predictions.columns if '_predicted_score' in model_name}
    num_matches = predictions['match_id'].nunique() 

    for match_id in predictions['match_id'].unique():
        match_data = predictions[predictions['match_id'] == match_id]
        match_data_sorted = match_data.sort_values(
            by=[col for col in match_data.columns if '_predicted_score' in col], ascending=False
        )
        for model_name in percentage_error_per_model:
            match_data_sorted_model = match_data_sorted.sort_values(by='fantasy_score_total', ascending=False)
            top_11_true = match_data_sorted_model['fantasy_score_total'].iloc[:11].copy()
            match_data_sorted_model = match_data.sort_values(by=model_name, ascending=False)
            if len(top_11_true) > 0:
                top_11_true.iloc[0] = 2 
            if len(top_11_true) > 1:
                top_11_true.iloc[1]= 1.5 

            top_11_predicted = match_data_sorted_model['fantasy_score_total'].iloc[:11].copy()
            if len(top_11_predicted) > 0:
                top_11_predicted.iloc[0] = 2
            if len(top_11_predicted) > 1:
                top_11_predicted.iloc[1]= 1.5
            top_11_sum_true = top_11_true.sum()
            top_11_sum_predicted = top_11_predicted.sum()
            if top_11_sum_true != 0:
                percentage_error = abs(top_11_sum_predicted - top_11_sum_true)/abs(top_11_sum_true)*100
            else:
                percentage_error = 0 

            percentage_error_per_model[model_name] += percentage_error

    avg_percentage_error_per_model = {model_name: total_error / num_matches for model_name, total_error in percentage_error_per_model.items()}

    return avg_percentage_error_per_model

def filter_by_date(df, start_date, end_date):
    # Convert the 'start_date' column to datetime format
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Filter the dataframe based on the date range
    filtered_df = df[(df['start_date'] >= start_date) & (df['start_date'] <= end_date)]
    
    return filtered_df

def train_and_save_model(train_start_date, train_end_date):

    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_test.csv")) # all features
    df = pd.read_csv(file_path, index_col=False)
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')

    output_model_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "model_artifacts", f"Product_UI_f{train_start_date}.pkl"))

    columns = ['batting_average_n2', 'batting_average_n3', 'boundary_percentage_n3',
                'centuries_cumsum', 'half_centuries_cumsum', 'economy_rate_n1',
                'economy_rate_n2', 'economy_rate_n3', 'wickets_in_n2_matches','wickets_in_n3_matches',
                'bowling_average_n2', 'bowling_strike_rate_n2', 'fielding_points',
                'longterm_avg_runs', 'longterm_var_runs', 'longterm_avg_strike_rate',
                'longterm_avg_wickets_per_match', 'longterm_var_wickets_per_match',
                'longterm_avg_economy_rate', 'longterm_total_matches_of_type',
                'avg_fantasy_score_5', 'avg_fantasy_score_12', 'avg_fantasy_score_15',
                'avg_fantasy_score_25', 'α_bowler_score_n3', 'order_seen', 'bowling_style',
                'gini_coefficient', 'batter', 'wicketkeeper', 'bowler', 'allrounder',
                'batting_style_Left hand Bat', 'start_date', 'fantasy_score_total', 'match_id', 'player_id']

    df = df[columns]

    df = drop_zero_dominant_rows(df, threshold=0.9)

    df = filter_by_date(df, train_start_date, train_end_date)

    y_train = df['fantasy_score_total']
    X_train = df.drop(['fantasy_score_total', 'start_date'], axis=1)
    X_train.drop(['match_id', 'player_id'], axis=1, inplace=True)
    trained_models = train_models(X_train, y_train)

    pickle.dump(trained_models, open(output_model_path, 'wb'))

train_and_save_model('2000-01-01', '2024-06-30')