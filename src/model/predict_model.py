import os
import pickle
import pandas as pd
import numpy as np
from typing import final
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import StackingRegressor, RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor
import shap
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

current_dir = os.path.dirname(os.path.abspath(__file__))

def predict_scores_c(trained_models, columns, X_test):
    """
    Predict scores using a collection of trained models.
    This function takes a dictionary of trained models, a list of columns to use for prediction,
    and a test dataset. It returns a DataFrame containing the predicted scores from each model.
    Parameters:
    trained_models (dict): A dictionary where keys are model names and values are dictionaries 
                           containing the trained model under the key 'model'.
    columns (list): A list of column names to be used for prediction.
    X_test (pd.DataFrame): The test dataset containing the features.
    Returns:
    pd.DataFrame: A DataFrame containing the predicted scores from each model. Each column 
                  corresponds to the predicted scores from a model, named as '<model_name>_predicted_score'.
    """
    X_test = X_test[columns]
    test_data = pd.DataFrame()
    for model_name, model_info in trained_models.items():
        model = model_info['model'] 
        try:
            if hasattr(model, "predict_proba"):
                pred_scores = model.predict_proba(X_test)[:, 1] 
            else:
                pred_scores = model.predict(X_test) 
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            pred_scores = np.zeros(X_test.shape[0]) 
        test_data[model_name + '_predicted_score'] = pred_scores

    return test_data

def predictions_per_match_c(trained_models,columns, X_test, test):
    predictions = predict_scores_c(trained_models, columns, X_test)
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id'] = test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')

    return predictions, test_reset


def predict_scores_odi(trained_models, columns, X_test):
    # Ensure columns of X_test align with X_train columns
    X_test = X_test[columns]

    test_data = pd.DataFrame()

    # Loop through each model to predict scores
    for model_name, model_info in trained_models.items():
        model = model_info['model']  # Extract the trained model
        pred_scores = model.predict(X_test)  # Predict the scores

        # Store each model's predicted scores in the DataFrame
        test_data[model_name + '_predicted_score'] = pred_scores

    return test_data

def predictions_per_match_odi(trained_models, columns, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores_odi(trained_models, columns, X_test)

    # Reset indices of test and predictions for alignment
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    # Assign match_id and fantasy_score_total from test to predictions DataFrame
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')
    # predictions['match_type'] = test_reset.get('match_type')

    return predictions, test_reset

def preprocess_odi(X):
    X=X.fillna(0)
    return X

def predict_scores_test(trained_model, X_test):
    # Ensure columns of X_test align with X_train columns

    test_data = pd.DataFrame()

    # Predict scores using the trained stacking model
    pred_scores = trained_model.predict(X_test)  # Predict the scores

    # Store the predicted scores in the DataFrame
    test_data['predicted_score'] = pred_scores

    return test_data

def predictions_per_match_test(trained_models, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores_test(trained_models, X_test)

    # Reset indices of test and predictions for alignment
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    # Assign match_id and fantasy_score_total from test to predictions DataFrame
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')
    # predictions['match_type'] = test_reset.get('match_type')


    return predictions

def filter_by_date(df, start_date, end_date):
    # Convert the 'start_date' column to datetime format
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Filter the dataframe based on the date range
    filtered_df = df[(df['start_date'] >= start_date) & (df['start_date'] <= end_date)]
    
    return filtered_df

def one_hot_encode_t20(X, column_name):
    unique_values = np.unique(X[column_name])

    one_hot_dict = {}

    # Create a binary column for each unique value
    for unique_value in unique_values:
        one_hot_dict[f"{column_name}_{unique_value}"] = (X[column_name] == unique_value).astype(int)

    # Remove the original column and add new one-hot encoded columns
    X = X.drop(columns=[column_name])
    for col_name, col_data in one_hot_dict.items():
        X[col_name] = col_data

    return X

def preprocess_t20(X):
    X= one_hot_encode_t20(X,'gender')
    #drop categorical columns
    cols=['bowling_average_n1',
       'bowling_strike_rate_n1', 'bowling_average_n2',
       'bowling_strike_rate_n2', 'bowling_average_n3',
       'bowling_strike_rate_n3','α_bowler_score']
    X=X.drop(cols,axis=1)
    return X


def encode_playing_role_vectorized_t20(df, column='playing_role'):
    """
    Optimized function to encode the 'playing_role' column into multiple binary columns
    using vectorized operations.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column containing playing roles.

    Returns:
    - pd.DataFrame: A DataFrame with binary columns ['batter', 'wicketkeeper', 'bowler', 'allrounder'].
    """
    # Initialize new columns with zeros
    df['batter'] = 0
    df['wicketkeeper'] = 0
    df['bowler'] = 0
    df['allrounder'] = 0

    # Handle non-null playing_role by replacing NaN with "None" and converting to lowercase for consistency
    non_null_roles = df[column].fillna("None").str.lower()  # Convert to lowercase

    # Vectorized checks for roles (we check if role contains certain keywords in lowercase)
    df['batter'] += non_null_roles.str.contains("batter").astype(int)
    df['wicketkeeper'] += non_null_roles.str.contains("wicketkeeper").astype(int)
    df['bowler'] += non_null_roles.str.contains("bowler").astype(int)
    df['allrounder'] += non_null_roles.str.contains("allrounder").astype(int)

    # Handle the 'Allrounder' specification of "Batting" or "Bowling" (e.g., "Batting Allrounder")
    df['batter'] += non_null_roles.str.contains("allrounder.*batting").astype(int)
    df['bowler'] += non_null_roles.str.contains("allrounder.*bowling").astype(int)

    # Fill NaN values with 0 (important to handle NaN properly before converting to int)
    df['batter'] = df['batter'].fillna(0).astype(int)
    df['wicketkeeper'] = df['wicketkeeper'].fillna(0).astype(int)
    df['bowler'] = df['bowler'].fillna(0).astype(int)
    df['allrounder'] = df['allrounder'].fillna(0).astype(int)

    return df[['batter', 'wicketkeeper', 'bowler', 'allrounder']]

def predict_scores_t20(trained_models, X_test):
    # Ensure columns of X_test align with X_train columns
    # X_test = X_test[numeric_columns]

    test_data = pd.DataFrame()

    # Loop through each model to predict scores
    for model_name, model_info in trained_models.items():
        model = model_info['model']  # Extract the trained model
        pred_scores = model.predict(X_test)  # Predict the scores

        # Store each model's predicted scores in the DataFrame
        test_data['predicted_score'] = pred_scores

    return test_data

def predictions_per_match_t20(trained_models, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores_t20(trained_models, X_test)

    # Reset indices of test and predictions for alignment
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')
    return predictions, test_reset

def preprocessdf_t20(df):

    # Convert 'start_date' and 'end_date' columns to datetime format
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.sort_values(by='start_date').reset_index(drop=True)
    return df


def generate_predictions_t20(train_start_date, train_end_date, test_start_date, test_end_date):
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')
    combined_model_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts", f"Model_UI_{train_start}-{train_end}.pkl"))
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)
    trained_models = combined_models['t20']
    
    trained_modelscc = trained_models['trained_modelscc']
    trained_modelsrr = trained_models['trained_modelsrr']
    neural_weights = trained_models['neural_weights']

    # Recreate the neural network model and set the loaded weights 
    neural = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  
    ])
    neural.compile(optimizer='adam', loss='mse', metrics=['mae'])
    neural.set_weights(neural_weights)
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_t20.csv"))
    df = pd.read_csv(file_path, index_col=False)

    columns = ['start_date', 'player_id', 'match_id', 'match_type', 'playing_role',
               'batting_average_n1', 'strike_rate_n1', 'boundary_percentage_n1',
               'batting_average_n2', 'strike_rate_n2', 'boundary_percentage_n2',
               'batting_average_n3', 'strike_rate_n3', 'boundary_percentage_n3',
               'centuries_cumsum', 'half_centuries_cumsum', 'avg_runs_scored',
               'avg_strike_rate', 'avg_half_centuries', 'avg_centuries',
               'avg_rolling_ducks', 'strike_rotation_percentage',
               'avg_strike_rotation_percentage', 'conversion_30_to_50',
               'economy_rate_n1', 'economy_rate_n2', 'economy_rate_n3',
               'wickets_in_n_matches', 'total_overs_throwed', 'bowling_average_n1',
               'bowling_strike_rate_n1', 'bowling_average_n2',
               'bowling_strike_rate_n2', 'bowling_average_n3',
               'bowling_strike_rate_n3', 'CBR', 'CBR2', 'fielding_points',
               'four_wicket_hauls_n', 'highest_runs', 'highest_wickets',
               'order_seen_mode', 'longterm_avg_runs', 'longterm_var_runs',
               'longterm_avg_strike_rate', 'longterm_avg_wickets_per_match',
               'longterm_var_wickets_per_match', 'longterm_avg_economy_rate',
               'longterm_total_matches_of_type', 'avg_fantasy_score_1',
               'avg_fantasy_score_5', 'avg_fantasy_score_10', 'avg_fantasy_score_15',
               'avg_fantasy_score_20', 'rolling_ducks', 'rolling_maidens', 'gender',
               'α_batsmen_score', 'α_bowler_score', 'batsman_rating', 'bowler_rating',
               'fantasy_score_total', 'longterm_total_matches_of_type', 'avg_against_opposition', 'bowling_style','selected']
    df = df[columns]
    df = preprocess_t20(df)
    df[['batter', 'wicketkeeper', 'bowler', 'allrounder']] = encode_playing_role_vectorized_t20(df, 'playing_role')
    df.drop(['longterm_total_matches_of_type','playing_role'], axis=1, inplace=True)
    df = preprocessdf_t20(df)
    test_df= filter_by_date(df, test_start_date, test_end_date)
    if test_df.empty:
        print("No matches found in the given date range.")
        return
    test_df = preprocess_odi(test_df)



    x_test1 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id','selected','match_type'], axis=1)
    x_test2 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id','match_type', 'selected'], axis=1)
        
    columns1 = x_test1.columns.to_list()
    columns2 = x_test2.columns.to_list()


    predictions_c, _ = predictions_per_match_c(trained_modelscc, columns1,x_test1, test_df)
    predictions_r, _ = predictions_per_match_odi(trained_modelsrr,columns2, x_test2, test_df)
    predictions=pd.merge(predictions_r,predictions_c,on=['match_id','player_id','fantasy_score_total'])
    X_test_nn= predictions.drop(columns=['match_id', 'player_id', 'fantasy_score_total'])
    predictions['my_predicted_score']=neural.predict(X_test_nn)
    predictions['predicted_score'] = predictions['my_predicted_score'].rename('predicted_score')
    predictions.drop(columns=["my_predicted_score","stacked_regression_model_predicted_score","stacked_classifier_model_predicted_score"],inplace=True)
    predictions = predictions[['predicted_score', 'match_id', 'player_id', 'fantasy_score_total']]
    output_file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "predictions_t20.csv"))
    predictions.to_csv(output_file_path, index=False)


def preprocess_test(X):
    X=X.fillna(0)
    return X

def generate_predictions_test(train_start_date, train_end_date,test_start_date, test_end_date):
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')
    combined_model_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts", f"Model_UI_{train_start}-{train_end}.pkl"))
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)
    trained_models = combined_models['test']
    
    trained_modelscc = trained_models['trained_modelscc']
    trained_modelsrr = trained_models['trained_modelsrr']
    neural_weights = trained_models['neural_weights']

    # Recreate the neural network model and set the loaded weights 
    neural = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])
    neural.compile(optimizer='adam', loss='mse', metrics=['mae'])
    neural.set_weights(neural_weights)
    
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_test.csv"))
    df = pd.read_csv(file_path, index_col=False)

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
            'batting_style_Left hand Bat', 'start_date', 'fantasy_score_total', 'match_id', 'player_id','selected']
    
    test_df = filter_by_date(df, test_start_date, test_end_date)
    if test_df.empty:
        print("No matches found in the given date range.")
        return
    test_df = test_df[columns]
    test_df = preprocess_test(test_df)



    x_test1 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id','selected'], axis=1)
    x_test2 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id', 'selected'], axis=1)
        
    columns1 = x_test1.columns.to_list()
    columns2 = x_test2.columns.to_list()


    predictions_c, _ = predictions_per_match_c(trained_modelscc, columns1,x_test1, test_df)
    predictions_r, _ = predictions_per_match_odi(trained_modelsrr,columns2, x_test2, test_df)
    predictions=pd.merge(predictions_r,predictions_c,on=['match_id','player_id','fantasy_score_total'])
    X_test_nn= predictions.drop(columns=['match_id', 'player_id', 'fantasy_score_total'])
    predictions['my_predicted_score']=neural.predict(X_test_nn)
    predictions['predicted_score'] = predictions['my_predicted_score'].rename('predicted_score')
    predictions.drop(columns=["my_predicted_score","stacked_regression_model_predicted_score","stacked_classifier_model_predicted_score"],inplace=True)
    predictions = predictions[['predicted_score', 'match_id', 'player_id', 'fantasy_score_total']]
    output_file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "predictions_test.csv"))
    predictions.to_csv(output_file_path, index=False)

def generate_predictions_odi(train_start_date, train_end_date,test_start_date, test_end_date):
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')
    combined_model_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts", f"Model_UI_{train_start}-{train_end}.pkl"))
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)
    trained_models = combined_models['odi']
    
    trained_modelscc = trained_models['trained_modelscc']
    trained_modelsrr = trained_models['trained_modelsrr']
    neural_weights = trained_models['neural_weights']

    # Recreate the neural network model and set the loaded weights 
    neural = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])
    neural.compile(optimizer='adam', loss='mse', metrics=['mae'])
    neural.set_weights(neural_weights)

    columns = [
       'player_id', 'match_id', 'match_type', 'start_date',
       'batting_average_n1', 'strike_rate_n1', 'boundary_percentage_n1',
       'batting_average_n2', 'strike_rate_n2', 'boundary_percentage_n2',
       'batting_average_n3', 'strike_rate_n3', 'boundary_percentage_n3',
       'centuries_cumsum', 'half_centuries_cumsum', 'avg_runs_scored',
       'avg_strike_rate', 'avg_half_centuries', 'avg_centuries',
       'avg_rolling_ducks', 'strike_rotation_percentage',
       'avg_strike_rotation_percentage', 'conversion_30_to_50',
       'economy_rate_n1', 'economy_rate_n2', 'economy_rate_n3',
       'wickets_in_n_matches', 'total_overs_throwed', 'CBR', 'CBR2', 'fielding_points',
       'four_wicket_hauls_n', 'highest_runs', 'highest_wickets',
       'order_seen_mode', 'longterm_avg_runs', 'longterm_var_runs',
       'longterm_avg_strike_rate', 'longterm_avg_wickets_per_match',
       'longterm_var_wickets_per_match', 'longterm_avg_economy_rate',
       'avg_fantasy_score_1', 'avg_fantasy_score_5', 'avg_fantasy_score_10', 'avg_fantasy_score_15',
       'avg_fantasy_score_20', 'rolling_ducks', 'rolling_maidens',
       'α_batsmen_score', 'batsman_rating', 'bowler_rating', 
       'fantasy_score_total', 'opponent_avg_fantasy_batting', 'opponent_avg_fantasy_bowling', 'avg_against_opposition', 'bowling_style', 'selected', 'home_away_away',
       'home_away_home', 'home_away_neutral', 'gender_female', 'gender_male', 'dot_ball_percentage_n1', 'dot_ball_percentage_n2', 'dot_ball_percentage_n3', 'longterm_dot_ball_percentage', 'dot_ball_percentage', 'longterm_var_dot_ball_percentage',
       'Pitch_Type_Batting-Friendly', 'role_factor', 'odi_impact',
       'Pitch_Type_Bowling-Friendly', 'Pitch_Type_Neutral', 'ARPO_venue',
       'BSR_venue']
 
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_odi.csv"))
    df = pd.read_csv(file_path, index_col=False)

    test_df = filter_by_date(df, test_start_date, test_end_date)
    if test_df.empty:
        print("No matches found in the given date range.")
        return
    test_df = test_df[columns]
    test_df = preprocess_odi(test_df)



    x_test1 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id','selected','match_type'], axis=1)
    x_test2 = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id','match_type', 'selected'], axis=1)
        
    columns1 = x_test1.columns.to_list()
    columns2 = x_test2.columns.to_list()


    predictions_c, _ = predictions_per_match_c(trained_modelscc, columns1,x_test1, test_df)
    predictions_r, _ = predictions_per_match_odi(trained_modelsrr,columns2, x_test2, test_df)
    predictions=pd.merge(predictions_r,predictions_c,on=['match_id','player_id','fantasy_score_total'])
    X_test_nn= predictions.drop(columns=['match_id', 'player_id', 'fantasy_score_total'])
    predictions['my_predicted_score']=neural.predict(X_test_nn)
    predictions['predicted_score'] = predictions['my_predicted_score'].rename('predicted_score')
    predictions.drop(columns=["my_predicted_score","stacked_regression_model_predicted_score","stacked_classifier_model_predicted_score"],inplace=True)
    output_file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "predictions_odi.csv"))
    predictions = predictions[['predicted_score', 'match_id', 'player_id', 'fantasy_score_total']]
    predictions.to_csv(output_file_path, index=False)

def main_generate_predictions(train_start, train_end, test_start, test_end):
    generate_predictions_t20(train_start, train_end, test_start, test_end)
    generate_predictions_odi(train_start, train_end, test_start, test_end)
    generate_predictions_test(train_start, train_end, test_start, test_end)