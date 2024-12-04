import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

current_dir = os.path.dirname(os.path.abspath(__file__))

def drop_zero_dominant_rows_odi(df, threshold=0.9):
    '''
    Drops rows from the input DataFrame where the percentage of zero values is greater than the threshold.
    '''
    zero_count = (df == 0).sum(axis=1)

    threshold_zeros = int(threshold * df.shape[1])

    df_cleaned = df[zero_count <= threshold_zeros]

    return df_cleaned

def train_models_regression(X_train, y_train):
    '''
    Trains regression models on the input data.
    '''
    models = {
    "xgboost regressor": XGBRegressor(random_state=42),
    "linear regression":LinearRegression(),
    "Catboost regressor":CatBoostRegressor(random_state=42,verbose=False),
}
    trained_models = {}

    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)

    if len(y_train.shape) == 1:
        pass
    elif len(y_train.shape) == 2:
        y_train = y_train.ravel()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = {
            'model': model,
        }
        
    return trained_models


def preprocess_odi(X):
    '''
    Preprocesses the input data for the ODI model.
    '''
    X=X.fillna(0)
    cols=['player_id','start_date','match_id','match_type']
    X=X.drop(cols,axis=1)
    return X


def train_models_classification(X_train, y_train):
    '''
    Trains classification models on the input data.
    '''
    models = {
        "xgboost classification": XGBClassifier(random_state=42),
        "logistic regression": LogisticRegression(),
        "Catboost classification": CatBoostClassifier(random_state=42, verbose=False),
    }
    
    trained_models = {}
    if isinstance(X_train, pd.Series):
        X_train = X_train.to_frame()  
    
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze() 

    if X_train.isnull().any().any() or not np.isfinite(X_train).all().all():
        raise ValueError("X_train contains NaN or infinite values.")
    if y_train.isnull().any() or not np.isfinite(y_train).all():
        raise ValueError("y_train contains NaN or infinite values.")
    
    for name, model in models.items():
        try:
            print(f"Training {name} with X_train shape {X_train.shape} and y_train shape {y_train.shape}...")
            model.fit(X_train, y_train)
            trained_models[name] = {'model': model}
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return trained_models


def predict_scores_c(trained_models, X_train, X_test):
    '''
    Takes trained models and input data and returns the predicted scores.
    '''
    X_test = X_test[X_train.columns]

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

def predictions_per_match_c(trained_models, X_train, X_test, test):
    '''
    Takes trained models, input data, and test data and returns the predictions.
    '''
    predictions = predict_scores_c(trained_models, X_train, X_test)

    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id'] = test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')

    return predictions, test_reset


def predict_scores_odi(trained_models, X_train, X_test):
    '''
    Takes trained models and input data and returns the predicted scores.
    '''
    X_test = X_test[X_train.columns]

    test_data = pd.DataFrame()

    for model_name, model_info in trained_models.items():
        model = model_info['model']
        pred_scores = model.predict(X_test)
        test_data[model_name + '_predicted_score'] = pred_scores

    return test_data

def predictions_per_match_odi(trained_models, X_train, X_test, test):
    '''
    Takes trained models, input data, and test data and returns the predictions.
    '''
    predictions = predict_scores_odi(trained_models, X_train, X_test)

    test_reset = test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    predictions['match_id'] = test_reset.get('match_id')
    predictions['player_id']=test_reset.get('player_id')
    predictions['fantasy_score_total'] = test_reset.get('fantasy_score_total')

    return predictions, test_reset

def train_neural_network(X, y):
    '''
    Trains a neural network on the input data.
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X, y, epochs=30, batch_size=32, verbose=1)
    return model

def filter_by_date(merged_df, start_date, end_date): 
    '''
    Filters the input DataFrame by the start and end dates.
    '''
    merged_df['start_date'] = pd.to_datetime(merged_df['start_date'])
    
    filtered_df = merged_df[(merged_df['start_date'] >= start_date) & (merged_df['start_date'] <= end_date)]
    
    return filtered_df


def iterative_training(X_train, y, X_trainc, yc, test):
    '''
    Trains models iteratively on the input data.
    '''
    final_predictions = []
    train_step_size = int(0.25*len(X_train))
    prediction_size = int(0.09*len(X_train))
    end = len(X_train)
    count = 0
    train_start = 0

    X_train_cumulative = []
    y_cumulative = []
    X_trainc_cumulative = []
    yc_cumulative = []
    count=0


    while train_start < end:
        train_end = min(train_start + train_step_size, end)
        print(f"train_start: {train_start} \n train_end: {train_end}")
        count += 1
        train_end = min(train_start + train_step_size,end)
        if train_end > end:
            break

        prediction_start = train_end
        prediction_end = prediction_start + prediction_size
        if prediction_start >= end:
            prediction_start = end
        if prediction_end > end:
            prediction_end = end

        X_train_batch = X_train[train_start:train_end]
        y_batch = y[train_start:train_end]
        X_trainc_batch = X_trainc[train_start:train_end]
        yc_batch = yc[train_start:train_end]

        X_train_cumulative.append(X_train_batch)
        y_cumulative.append(y_batch)
        X_trainc_cumulative.append(X_trainc_batch)
        yc_cumulative.append(yc_batch)

        X_train_combined = pd.concat(X_train_cumulative, ignore_index=True)
        y_combined = pd.concat(y_cumulative, ignore_index=True)
        X_trainc_combined = pd.concat(X_trainc_cumulative, ignore_index=True)
        yc_combined = pd.concat(yc_cumulative, ignore_index=True)

        X_pred_batch = X_train[prediction_start:prediction_end]
        X_predc_batch = X_trainc[prediction_start:prediction_end]

        trained_modelsr = train_models_regression(X_train_combined, y_combined)
        trained_modelsc = train_models_classification(X_trainc_combined, yc_combined)

        if(prediction_end-prediction_start>0):
            predictions_r, _ = predictions_per_match_odi(
                trained_modelsr,
                X_train_combined,
                X_pred_batch,
                test.iloc[prediction_start:prediction_end],
            )
            predictions_c, _ = predictions_per_match_c(
                trained_modelsc,
                X_trainc_combined,
                X_predc_batch,
                test.iloc[prediction_start:prediction_end],
            )

            predictions = pd.merge(
                predictions_r,
                predictions_c,
                on=['match_id', 'player_id', 'fantasy_score_total'],
                how='inner',
                validate='one_to_one'
            )

            final_predictions.append(predictions)

        train_start += train_step_size

    combined_predictions = pd.concat(final_predictions, ignore_index=True)

    return trained_modelsr, trained_modelsc, combined_predictions

def drop_zero_dominant_rows_test(df, threshold=0.9):
    '''
    Drops rows from the input DataFrame where the percentage of zero values is greater than the threshold.
    '''
    zero_count = (df == 0).sum(axis=1)

    threshold_zeros = int(threshold * df.shape[1]) 

    df_cleaned = df[zero_count <= threshold_zeros]

    return df_cleaned

def train_models_test(X_train, y_train):
    base_models = [
        ("linear_regression", LinearRegression()),
        ('catboost', CatBoostRegressor(verbose=0, random_state=42)),
        ("mlp", MLPRegressor(random_state=42, max_iter=1000))
    ]

    meta_model = LGBMRegressor(random_state = 42)

    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)

    if len(y_train.shape) == 1: 
        pass
    elif len(y_train.shape) == 2:
        y_train = y_train.ravel()

    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacked_model.fit(X_train, y_train)

    return stacked_model

def one_hot_encode_t20(X, column_name):
    '''
    One-hot encodes the input column in the input DataFrame.
    '''
    unique_values = np.unique(X[column_name])

    one_hot_dict = {}

    for unique_value in unique_values:
        one_hot_dict[f"{column_name}_{unique_value}"] = (X[column_name] == unique_value).astype(int)

    X = X.drop(columns=[column_name])
    for col_name, col_data in one_hot_dict.items():
        X[col_name] = col_data

    return X

def preproces_t20(X):
    '''
    Preprocesses the input data for the T20 model.
    '''
    X= one_hot_encode_t20(X,'gender')
    cols=['player_id','bowling_average_n1',
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
    df['batter'] = 0
    df['wicketkeeper'] = 0
    df['bowler'] = 0
    df['allrounder'] = 0

    non_null_roles = df[column].fillna("None").str.lower()

    df['batter'] += non_null_roles.str.contains("batter").astype(int)
    df['wicketkeeper'] += non_null_roles.str.contains("wicketkeeper").astype(int)
    df['bowler'] += non_null_roles.str.contains("bowler").astype(int)
    df['allrounder'] += non_null_roles.str.contains("allrounder").astype(int)

    df['batter'] += non_null_roles.str.contains("allrounder.*batting").astype(int)
    df['bowler'] += non_null_roles.str.contains("allrounder.*bowling").astype(int)

    df['batter'] = df['batter'].fillna(0).astype(int)
    df['wicketkeeper'] = df['wicketkeeper'].fillna(0).astype(int)
    df['bowler'] = df['bowler'].fillna(0).astype(int)
    df['allrounder'] = df['allrounder'].fillna(0).astype(int)

    return df[['batter', 'wicketkeeper', 'bowler', 'allrounder']]

def preprocessdf_t20(df):
    '''
    Preprocesses the input data for the T20 model.
    '''
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.sort_values(by='start_date').reset_index(drop=True)
    return df

def train_models_t20(X_train, y_train):
    models = {
        "linear regression": LinearRegression(),
        "ridge regression": Ridge(),
        "lasso regression": Lasso(),
        "elastic net": ElasticNet(),
        "Catboost regressor": CatBoostRegressor(random_state=42, verbose=False),
         "xgboost regressor": XGBRegressor(random_state=42)
    }
    
    trained_models = {}
    
    if isinstance(X_train, pd.Series):
        X_train = X_train.to_frame()
    
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()

    if X_train.isnull().any().any() or not np.isfinite(X_train).all().all():
        raise ValueError("X_train contains NaN or infinite values.")
    if y_train.isnull().any() or not np.isfinite(y_train).all():
        raise ValueError("y_train contains NaN or infinite values.")
    
    for name, model in models.items():
        try:
            print(f"Training {name} with X_train shape {X_train.shape} and y_train shape {y_train.shape}...")
            model.fit(X_train, y_train)
            trained_models[name] = {'model': model}
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return trained_models

 
def train_and_save_model_t20(train_start_date, train_end_date):
    '''
    Trains the T20 model on the input data and saves the trained model to a file.
    '''
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    model_output_path = this_file_dir + '../model_artifacts/Model_UI_' + train_start + '-' + train_end + '_t20' + '.pkl' 
    features_t20_path = this_file_dir + '../data/processed/final_training_file_t20.csv'

    df = pd.read_csv(features_t20_path, index_col=False)
    columns=['start_date','player_id', 'match_id', 'match_type','playing_role',
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
        'avg_fantasy_score_20', 'rolling_ducks', 'rolling_maidens','gender',
        'α_batsmen_score', 'α_bowler_score', 'batsman_rating', 'bowler_rating',
        'fantasy_score_total','longterm_total_matches_of_type','avg_against_opposition','bowling_style']
    df = df[columns]
    df = preproces_t20(df)
    df[['batter', 'wicketkeeper', 'bowler', 'allrounder']] = encode_playing_role_vectorized_t20(df, 'playing_role')
    df.drop('longterm_total_matches_of_type', axis=1, inplace=True)
    df = preprocessdf_t20(df)
    train = filter_by_date(df, train_start_date, train_end_date)
    y_train = train['fantasy_score_total']
    train.drop(['match_type','match_id'], axis=1, inplace=True)
    train.fillna(0, inplace=True)
    numeric_X_train = train.select_dtypes(include=[np.number])
    trained_models = train_models_t20(numeric_X_train.drop('fantasy_score_total', axis=1), y_train)
    with open(model_output_path, 'wb') as file:
        pickle.dump(trained_models, file)

def train_and_save_model_test(train_start_date, train_end_date):

    file_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "processed", "final_training_file_test.csv")) # all features
    df = pd.read_csv(file_path, index_col=False)
    train_start = train_start_date.replace('-', '_')
    train_end = train_end_date.replace('-', '_')

    output_model_path = os.path.abspath(os.path.join(current_dir, "..", "..","src", "model_artifacts", f"Model_UI_{train_start}-{train_end}_test.pkl"))

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

    df = drop_zero_dominant_rows_test(df, threshold=0.9)

    df = filter_by_date(df, train_start_date, train_end_date)

    y_train = df['fantasy_score_total']
    X_train = df.drop(['fantasy_score_total', 'start_date'], axis=1)
    X_train.drop(['match_id', 'player_id'], axis=1, inplace=True)
    trained_models = train_models_test(X_train, y_train)

    pickle.dump(trained_models, open(output_model_path, 'wb'))

def train_and_save_model_odi(train_start_date, train_end_date):
    cols = [
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
        'BSR_venue'
    ]

    train_start = train_start_date.replace("-", "_")
    train_end = train_end_date.replace("-", "_")
    model_output_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts" , f"Model_UI_{train_start}-{train_end}_odi.pkl"))
    features_odi_path = os.path.abspath(os.path.join(current_dir, "..", "data" , "processed" , "final_training_file_odi.csv"))

    df = pd.read_csv(features_odi_path, index_col=False)

    df = df[cols]

    df = drop_zero_dominant_rows_odi(df)

    train = filter_by_date(df, train_start_date, train_end_date)

    y_train = train['fantasy_score_total']
    x_train = train.drop(['selected','fantasy_score_total'], axis=1)

    x_train = preprocess_odi(x_train)

    y_trainc = train['selected']
    x_trainc = train.drop(['fantasy_score_total', 'selected'], axis=1)

    x_trainc = preprocess_odi(x_trainc)

    shuffled_indices = np.random.permutation(train.index)

    X_train = x_train.loc[shuffled_indices].reset_index(drop=True)
    y_train = y_train.loc[shuffled_indices].reset_index(drop=True)
    X_trainc = x_trainc.loc[shuffled_indices].reset_index(drop=True)
    y_trainc = y_trainc.loc[shuffled_indices].reset_index(drop=True)
    train = train.loc[shuffled_indices].reset_index(drop=True)

    trained_modelsrr, trained_modelscc, combined = iterative_training(X_train, y_train, X_trainc, y_trainc, train)

    Xn = combined.drop(['match_id', 'player_id', 'fantasy_score_total'], axis=1)
    yn = combined['fantasy_score_total']

    neural = train_neural_network(Xn, yn)

    with open(model_output_path, 'wb') as file:
        pickle.dump({
            'trained_modelscc': trained_modelscc,
            'trained_modelsrr': trained_modelsrr,
            'neural_weights': neural.get_weights()
        }, file)

def model_merge(train_start_date, train_end_date):
    '''
    Merges the ODI, Test, and T20 models into a single model.
    '''
    train_date = train_start_date.replace("-", "_")
    end_date = train_end_date.replace("-", "_")

    model_odi_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts" , f"Model_UI_{train_date}-{end_date}_odi.pkl"))
    model_test_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts" , f"Model_UI_{train_date}-{end_date}_test.pkl"))
    model_t20_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts" , f"Model_UI_{train_date}-{end_date}_t20.pkl"))

    model_odi = pickle.load(open(model_odi_path, 'rb'))
    model_test = pickle.load(open(model_test_path, 'rb'))
    model_t20 = pickle.load(open(model_t20_path, 'rb'))

    combined_models = {
        'odi': model_odi,
        'test': model_test,
        't20': model_t20
    }


    combined_model_path = os.path.abspath(os.path.join(current_dir, "..", "model_artifacts" , f"Model_UI_{train_date}-{end_date}.pkl"))
    pickle.dump(combined_models, open(combined_model_path, 'wb'))
    os.remove(model_odi_path)
    os.remove(model_test_path)
    os.remove(model_t20_path)


def main_train_and_save(start,end):
    train_and_save_model_odi(start, end)
    train_and_save_model_test(start, end)
    train_and_save_model_t20(start, end)
    model_merge(start, end)