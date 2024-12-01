import pandas as pd
import os
import pickle

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

def get_recent_features(player_ids, match_date, feature_csv_path):
    """
    Extracts the most recent features for given player_ids before a specified match_date.

    Args:
        player_ids (list): List of player IDs to filter.
        match_type (str): Match type to filter.
        match_date (str): Match date (YYYY-MM-DD) for comparison.
        feature_csv_path (str): Path to the feature CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the most recent features for the specified players.
    """
    df_features = pd.read_csv(feature_csv_path)
    df_features['player_id'] = df_features['player_id'].astype(str)
    match_date = pd.to_datetime(match_date)
    df_features['start_date'] = pd.to_datetime(df_features['start_date'])
    
    # Filter data based on match_type and player_ids
    filtered_df = df_features[
        (df_features['player_id'].isin(player_ids)) & 
        # (df_features['match_type'] == match_type) & 
        (df_features['start_date'] < match_date)
    ]

    # Find the most recent entry for each player_id
    most_recent_df = filtered_df.sort_values(by=['player_id', 'start_date'], ascending=[True, False]) \
                                .groupby('player_id').first().reset_index()
    
    return most_recent_df

feature_csv_path = os.path.abspath(os.path.join(current_dir,"src", "data", "processed", "final_training_file_test.csv"))
 
model_path = os.path.abspath(os.path.join(current_dir,"src", "model_artifacts",f"Product_UI_f2000-01-01.pkl" ))
with open(model_path, 'rb') as file:
        trained_models = pickle.load(file)
 
def get_player_scores(match_type , player_ids , match_date):
    recent_features_df = get_recent_features(player_ids, match_date, feature_csv_path)
    recent_features_df = recent_features_df[columns]
    x_test = recent_features_df.drop(['fantasy_score_total', 'start_date', 'match_id', 'player_id'], axis=1)
    predictions = predictions_per_match(trained_models, x_test, recent_features_df)
    player_ids = predictions['player_id']
    predicted_scores = predictions['stacked_model_predicted_score']
    player_score = dict(zip(player_ids, predicted_scores))
    return player_score

match_type = 'Test'
player_score = get_player_scores(match_type ,["0085a7ce", "00823a96"], "2024-12-01")
print(player_score)

