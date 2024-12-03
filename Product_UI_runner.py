import pandas as pd
import os
import pickle
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
    #X= one_hot_encode(X,'match_type')
    # X= one_hot_encode(X,'playing_role')
    # X= one_hot_encode(X,'bowling_style')
    # X= one_hot_encode(X,'batting_style')
    X= one_hot_encode_t20(X,'gender')
    # X=X.fillna(0)
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

    # Assign match_id and fantasy_score_total from test to predictions DataFrame
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')
    # predictions['match_type'] = test_reset.get('match_type')

    return predictions, test_reset

def preprocessdf_t20(df):
    # df['start_date']= df['date']
    # df.drop('date', axis=1, inplace=True)
    # df.drop('end_date', axis=1, inplace=True)

    # Convert 'start_date' and 'end_date' columns to datetime format
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.sort_values(by='start_date').reset_index(drop=True)
    return df

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

def preprocess_odi(X):
    X=X.fillna(0)
    return X

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

def predictions_per_match_c(trained_models,columns, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores_c(trained_models, columns, X_test)

    # Reset indices of test and predictions for alignment
    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    # Assign match_id, player_id, and fantasy_score_total from test to predictions DataFrame
    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id'] = test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')

    return predictions, test_reset

def predict_scores_c(trained_models, columns, X_test):
    # Ensure columns of X_test align with X_train columns
    X_test = X_test[columns]

    test_data = pd.DataFrame()

    # Loop through each model to predict scores
    for model_name, model_info in trained_models.items():
        model = model_info['model']  # Extract the trained model
        
        # Predict probabilities or binary outcomes based on model's capabilities
        try:
            # Predict probabilities if available, otherwise predict binary labels
            if hasattr(model, "predict_proba"):
                pred_scores = model.predict_proba(X_test)[:, 1]  # Use probability for class 1
            else:
                pred_scores = model.predict(X_test)  # Use binary labels
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            pred_scores = np.zeros(X_test.shape[0])  # Default to zero scores if prediction fails

        # Store each model's predicted scores in the DataFrame
        test_data[model_name + '_predicted_score'] = pred_scores

    return test_data

def generate_predictions_odi(test_df):
    # # Load the trained models
    # model_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "model_artifacts",f"Model_UI_{train_start}-{train_end}_odi.pkl" ))
    # with open(model_path, 'rb') as file:
    #     trained_models = pickle.load(file)
    combined_model_path = os.path.abspath(os.path.join(current_dir, "src","model_artifacts", "Product_UI_2000_01_01-2024_06_30.pkl"))
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)

    trained_models = combined_models['odi']
    
    trained_modelscc = trained_models['trained_modelscc']
    trained_modelsrr = trained_models['trained_modelsrr']
    neural_weights = trained_models['neural_weights']

    # Recreate the neural network model and set the loaded weights 
    neural = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])
    neural.compile(optimizer='adam', loss='mse', metrics=['mae'])
    neural.set_weights(neural_weights)

    columns_odi = [
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
       'fantasy_score_total', 'bowling_style', 'selected', 
    'gender_female', 'gender_male', 'dot_ball_percentage_n1', 'dot_ball_percentage_n2', 'dot_ball_percentage_n3', 'longterm_dot_ball_percentage', 'dot_ball_percentage', 'longterm_var_dot_ball_percentage',
         'role_factor', 'odi_impact']

    test_df = test_df[columns_odi]
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
    predictions.drop(columns=["my_predicted_score","xgboost regressor_predicted_score","linear regression_predicted_score","Catboost regressor_predicted_score","xgboost classification_predicted_score","logistic regression_predicted_score","Catboost classification_predicted_score"],inplace=True)
    predictions = predictions[[ 'player_id','predicted_score']]
    # convert predictions to a list 
    predictions = predictions.to_dict(orient='records')
    return predictions



def generate_predictions_test(test_df):
    combined_model_path = os.path.abspath(os.path.join(current_dir, "src", "model_artifacts", "Product_UI_2000_01_01-2024_06_30.pkl"))
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)
    trained_models = combined_models['test']

    columns_test = ['batting_average_n2', 'batting_average_n3', 'boundary_percentage_n3',
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
    test_df = test_df[columns_test]
    x_test = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id'], axis=1)
    predictions = predictions_per_match_test(trained_models, x_test, test_df)
    predictions = predictions[[ 'player_id','predicted_score']]
    predictions = predictions.to_dict(orient='records')
    return predictions

def generate_predictions_t20(test_df):
    combined_model_path = os.path.abspath(os.path.join(current_dir, "src", "model_artifacts", "Product_UI_2000_01_01-2024_06_30.pkl"))

    columns_t20 = ['start_date', 'player_id', 'match_id', 'match_type', 'playing_role',
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
               'fantasy_score_total', 'longterm_total_matches_of_type','bowling_style']

    numeric_columns_t20 = ['batting_average_n1', 'strike_rate_n1', 'boundary_percentage_n1',
                       'batting_average_n2', 'strike_rate_n2', 'boundary_percentage_n2',
                       'batting_average_n3', 'strike_rate_n3', 'boundary_percentage_n3',
                       'centuries_cumsum', 'half_centuries_cumsum', 'avg_runs_scored',
                       'avg_strike_rate', 'avg_half_centuries', 'avg_centuries',
                       'avg_rolling_ducks', 'strike_rotation_percentage',
                       'avg_strike_rotation_percentage', 'conversion_30_to_50',
                       'economy_rate_n1', 'economy_rate_n2', 'economy_rate_n3',
                       'wickets_in_n_matches', 'total_overs_throwed', 'CBR', 'CBR2',
                       'fielding_points', 'four_wicket_hauls_n', 'highest_runs',
                       'highest_wickets', 'order_seen_mode', 'longterm_avg_runs',
                       'longterm_var_runs', 'longterm_avg_strike_rate',
                       'longterm_avg_wickets_per_match', 'longterm_var_wickets_per_match',
                       'longterm_avg_economy_rate', 'avg_fantasy_score_1',
                       'avg_fantasy_score_5', 'avg_fantasy_score_10', 'avg_fantasy_score_15',
                       'avg_fantasy_score_20', 'rolling_ducks', 'rolling_maidens',
                       'α_batsmen_score', 'batsman_rating', 'bowler_rating',
                         'bowling_style',
                       'gender_female', 'gender_male', 'batter', 'wicketkeeper', 'bowler',
                       'allrounder']

    test_df = test_df[columns_t20]
    test_df = preprocess_t20(test_df)
    test_df[['batter', 'wicketkeeper', 'bowler', 'allrounder']] = encode_playing_role_vectorized_t20(test_df, 'playing_role')
    test_df.drop('longterm_total_matches_of_type', axis=1, inplace=True)
    test_df = preprocessdf_t20(test_df)
    test_df.drop(['match_type'], axis=1, inplace=True)
    with open(combined_model_path, 'rb') as file:
        combined_models = pickle.load(file)

    trained_models = combined_models['t20']

    X_test = test_df[numeric_columns_t20]

    X_test.fillna(0, inplace=True)

    predictions, _ = predictions_per_match_t20(trained_models, X_test, test_df)

     
    predictions = predictions[['player_id', 'predicted_score']]
    predictions = predictions.to_dict(orient='records')
    return predictions



def main(match_type, player_list, match_date):
    match_type = match_type.upper()
    final_training_file_ODI = pd.read_csv(os.path.abspath(os.path.join(current_dir, "src","data","processed", "final_training_file_ODI.csv")), index_col=False)
    final_training_file_TEST = pd.read_csv(os.path.abspath(os.path.join(current_dir, "src","data","processed", "final_training_file_test.csv")),index_col=False)
    final_training_file_T20 = pd.read_csv(os.path.abspath(os.path.join(current_dir, "src","data","processed", "final_training_file_t20.csv")),index_col=False)

    if(match_type=='ODI' or match_type=='ODM'):

        testing_data = pd.DataFrame()

        for player_id in player_list:   
            filtered_data = final_training_file_ODI[final_training_file_ODI['start_date'] < match_date]
            recent_entry = filtered_data[filtered_data['player_id'] == player_id].sort_values(by='start_date', ascending=False).groupby('player_id').first().reset_index()
            
            if recent_entry.empty:
               recent_entry = pd.DataFrame([len(final_training_file_ODI.columns)*[0]], columns=final_training_file_ODI.columns)
               recent_entry['player_id'] = player_id

            
            testing_data = pd.concat([testing_data, recent_entry], axis=0) 
        testing_data.to_csv('testing_data.csv')
        odi_scores = generate_predictions_odi(testing_data)
        return odi_scores
    
    elif(match_type=='TEST' or match_type=='MDM'):

        testing_data = pd.DataFrame()

        for player_id in player_list:
            filtered_data = final_training_file_TEST[final_training_file_TEST['start_date'] < match_date]
            recent_entry = filtered_data[filtered_data['player_id'] == player_id].sort_values(by='start_date', ascending=False).groupby('player_id').first().reset_index()
            
            if recent_entry.empty:
                recent_entry = pd.DataFrame([len(final_training_file_TEST.columns)*[0]], columns=final_training_file_TEST.columns)
                recent_entry['player_id'] = player_id

            
            testing_data = pd.concat([testing_data, recent_entry], axis=0)

        test_scores = generate_predictions_test(testing_data)
        return test_scores
    
    elif(match_type=='T20' or match_type=='IT20'):

        testing_data = pd.DataFrame()

        for player_id in player_list:
            filtered_data = final_training_file_T20[final_training_file_T20['start_date'] < match_date]
            recent_entry = filtered_data[filtered_data['player_id'] == player_id].sort_values(by='start_date', ascending=False).groupby('player_id').first().reset_index()
            if recent_entry.empty:
               recent_entry = pd.DataFrame([len(final_training_file_T20.columns)*[0]], columns=final_training_file_T20.columns)
               recent_entry['player_id'] = player_id

            testing_data = pd.concat([testing_data, recent_entry], axis=0)

        t20_scores = generate_predictions_t20(testing_data)
        return t20_scores
     
player_list = [
    "f44af07f",
    "8b5b6769",
    "ddc0828d",
    "a4af4849",
    "0184dc35",
    "e68dd455",
    "8ba8195d",
    "725529bc",
    "bad31fac",
    "6e1b1ec0",
    "f76ffa81",
    "d2c2b2d5",
    "d1d0b5ed",
    "8fb88ee3",
    "dc0f5506",
    "96d3958e",
    "0c2730df",
    "2975a46c",
    "de8cce37",
    "ea42ddb9",
    "b6cd714f",
    "4bd33839"
]
match_date = "2010-12-19"
match_type = "odi"
scores = main(match_type, player_list, match_date)
with open('scores.txt', 'w') as f:
    for score in scores:
        f.write(f"{score}\n")

