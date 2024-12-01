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

def predict_scores(trained_model, X_test):
    # Ensure columns of X_test align with X_train columns

    test_data = pd.DataFrame()

    # Predict scores using the trained stacking model
    pred_scores = trained_model.predict(X_test)  # Predict the scores

    # Store the predicted scores in the DataFrame
    test_data['stacked_model_predicted_score'] = pred_scores

    return test_data

def predictions_per_match(trained_models, X_test, test):
    # Call predict_scores to get the predicted scores DataFrame
    predictions = predict_scores(trained_models, X_test)

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

def generate_predictions(train_start_date, train_end_date,test_start_date, test_end_date):
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')
    # Load the trained models
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "model_artifacts",f"Model_UI_{train_start}-{train_end}.pkl" ))
    with open(model_path, 'rb') as file:
        trained_models = pickle.load(file)
    
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_test.csv"))
    df = pd.read_csv(file_path, index_col=False)

    df = df[columns]
   

    test_df = filter_by_date(df, test_start_date, test_end_date)


    x_test = test_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id'], axis=1)
 

    predictions = predictions_per_match(trained_models, x_test, test_df)

    output_file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "predictions_test.csv"))
    predictions.to_csv(output_file_path, index=False)

