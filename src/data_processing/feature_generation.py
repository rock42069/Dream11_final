import pandas as pd
import numpy as np
import geonamescache
import pycountry
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_mw_pw = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "interim", "mw_pw_profiles.csv"))
file_path_mw_overall = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "interim", "mw_overall.csv"))

def rolling_dot_balls_features(group, n1=3, n2=7, n3=12):
    """
    Calculate bowling averages, economy rates, strike rates, updated CBR, 
    and fielding points using a rolling window.
    """
    group = group.sort_values('start_date')

    def calculate_rolling_metrics(group, n,min_periods,name):
        balls = group['balls_bowled'].shift().rolling(n, min_periods=min_periods).sum()
        group[name] = (group['dot_balls_as_bowler'].shift().rolling(n, min_periods=min_periods).sum() / balls)*100

        return group
    
    group = calculate_rolling_metrics(group, n1,1,'dot_ball_percentage_n1')
    group = calculate_rolling_metrics(group, n2,3,'dot_ball_percentage_n2')
    group = calculate_rolling_metrics(group, n3,5,'dot_ball_percentage_n3')

    return group


def longtermfeatures_dot_balls(group):
    """Calculate long-term dot ball_percentage"""
    group = group.sort_values('start_date')
    balls = group['balls_bowled'].shift().expanding().sum()
    group['longterm_dot_ball_percentage'] = (group['dot_balls_as_bowler'].shift().expanding().sum() / balls) * 100
    group['dot_ball_percentage'] = (group['dot_balls_as_bowler'] / group['balls_bowled']) * 100
    group['longterm_var_dot_ball_percentage'] = np.sqrt(group['dot_ball_percentage'].shift().expanding().var())

    return group

def calculate_centuries(runs_scored):
    """Calculate the total number of centuries."""
    return (runs_scored >= 100).sum()

def calculate_half_centuries(runs_scored):
    """Calculate the total number of half-centuries (50 <= runs < 100)."""
    return ((runs_scored < 100) & (runs_scored >= 50)).sum()

def calculate_rolling_batting_stats_test(group, n1=3, n2=7, n3=12,min_balls=20):
    """Calculate batting averages, strike rates, and boundary percentages using a rolling window."""
    group = group.sort_values('start_date')
    
    runs_n1 = group['runs_scored'].shift().rolling(n1, min_periods=1).sum()
    balls_n1 = group['balls_faced'].shift().rolling(n1, min_periods=1).sum()
    player_out_n1 = group['player_out'].shift().rolling(n1, min_periods=1).sum()
    boundary_runs_n1 = (group['fours_scored'].shift().rolling(n1, min_periods=1).sum() * 4 +
                        group['sixes_scored'].shift().rolling(n1, min_periods=1).sum() * 6)

    group['batting_average_n1'] = runs_n1 / player_out_n1.replace(0, np.nan)
    group['strike_rate_n1'] = np.where(balls_n1 >= min_balls, (runs_n1 / balls_n1) * 100, np.nan)
    group['boundary_percentage_n1'] = np.where(runs_n1 > 0, (boundary_runs_n1 / runs_n1) * 100, np.nan)

    runs_n2 = group['runs_scored'].shift().rolling(n2, min_periods=3).sum()
    balls_n2 = group['balls_faced'].shift().rolling(n2, min_periods=3).sum()
    player_out_n2 = group['player_out'].shift().rolling(n2, min_periods=3).sum()
    boundary_runs_n2 = (group['fours_scored'].shift().rolling(n2, min_periods=3).sum() * 4 +
                        group['sixes_scored'].shift().rolling(n2, min_periods=3).sum() * 6)

    group['batting_average_n2'] = runs_n2 / player_out_n2.replace(0, np.nan)
    group['strike_rate_n2'] = np.where(balls_n2 >= min_balls, (runs_n2 / balls_n2) * 100, np.nan)
    group['boundary_percentage_n2'] = np.where(runs_n2 > 0, (boundary_runs_n2 / runs_n2) * 100, np.nan)

    runs_n3 = group['runs_scored'].shift().rolling(n3, min_periods=5).sum()
    balls_n3 = group['balls_faced'].shift().rolling(n3, min_periods=5).sum()
    player_out_n3 = group['player_out'].shift().rolling(n3, min_periods=5).sum()
    boundary_runs_n3 = (group['fours_scored'].shift().rolling(n3, min_periods=5).sum() * 4 +
                        group['sixes_scored'].shift().rolling(n3, min_periods=5).sum() * 6)

    group['batting_average_n3'] = runs_n3 / player_out_n3.replace(0, np.nan)
    group['strike_rate_n3'] = np.where(balls_n3 >= min_balls, (runs_n3 / balls_n3) * 100, np.nan)
    group['boundary_percentage_n3'] = np.where(runs_n3 > 0, (boundary_runs_n3 / runs_n3) * 100, np.nan)

    return group

def calculate_rolling_bowling_stats_test(group, n1=3, n2=7, n3=12):
    """
    Calculate bowling averages, economy rates, strike rates, and an updated CBR using a rolling window.
    """
    group = group.sort_values('start_date')

    runs_n1 = group['runs_conceded'].shift().rolling(n1, min_periods=1).sum()
    wickets_n1 = group['wickets_taken'].shift().rolling(n1, min_periods=1).sum()
    balls_n1 = group['balls_bowled'].shift().rolling(n1, min_periods=1).sum()

    group['bowling_average_n1'] = runs_n1 / wickets_n1.replace(0, np.nan)
    group['economy_rate_n1'] = runs_n1 / (balls_n1 / group['balls_per_over'].iloc[0])
    group['bowling_strike_rate_n1'] = balls_n1 / wickets_n1.replace(0, np.nan)

    runs_n2 = group['runs_conceded'].shift().rolling(n2, min_periods=3).sum()
    wickets_n2 = group['wickets_taken'].shift().rolling(n2, min_periods=3).sum()
    balls_n2 = group['balls_bowled'].shift().rolling(n2, min_periods=3).sum()

    group['bowling_average_n2'] = runs_n2 / wickets_n2.replace(0, np.nan)
    group['economy_rate_n2'] = runs_n2 / (balls_n2 / group['balls_per_over'].iloc[0])
    group['bowling_strike_rate_n2'] = balls_n2 / wickets_n2.replace(0, np.nan)

    runs_n3 = group['runs_conceded'].shift().rolling(n3, min_periods=5).sum()
    wickets_n3 = group['wickets_taken'].shift().rolling(n3, min_periods=5).sum()
    balls_n3 = group['balls_bowled'].shift().rolling(n3, min_periods=5).sum()

    group['bowling_average_n3'] = runs_n3 / wickets_n3.replace(0, np.nan)
    group['economy_rate_n3'] = runs_n3 / (balls_n3 / group['balls_per_over'].iloc[0])
    group['bowling_strike_rate_n3'] = balls_n3 / wickets_n3.replace(0, np.nan)

    def calculate_cbr(avg, econ, sr):
        avg = np.where(avg > 0, np.log1p(avg), np.inf)
        econ = np.where(econ > 0, np.log1p(econ), np.inf)
        sr = np.where(sr > 0, np.log1p(sr), np.inf)

        return (avg * econ * sr) / (avg + econ + sr)

    group['CBR'] = calculate_cbr(
        group['bowling_average_n2'], group['economy_rate_n2'], group['bowling_strike_rate_n2']
    )

    group['fielding_points'] = (
        group['catches_taken'].shift().rolling(n3, min_periods=5).sum() * 8 +
        group['stumpings_done'].shift().rolling(n3, min_periods=5).sum() * 12 +
        group['run_out_direct'].shift().rolling(n3, min_periods=5).sum() * 12 +
        group['run_out_throw'].shift().rolling(n3, min_periods=5).sum() * 6
    )

    return group



def calculate_centuries_and_half_centuries(group):
    """Calculate cumulative centuries and half-centuries up to each date."""
    group = group.sort_values('start_date')
    group['centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(calculate_centuries)
    group['half_centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(calculate_half_centuries)
    return group

def calculate_additional_stats(group, n1=3,n2=7,n3=12):
    """Calculate additional cumulative and rolling stats for wickets and overs bowled."""
    group = group.sort_values('start_date')
    group[f'wickets_in_n1_matches'] = group['wickets_taken'].shift().rolling(n1, min_periods=1).sum()
    group[f'wickets_in_n2_matches'] = group['wickets_taken'].shift().rolling(n2, min_periods=3).sum()
    group[f'wickets_in_n3_matches'] = group['wickets_taken'].shift().rolling(n3, min_periods=5).sum()
    group[f'total_overs_throwed_n1'] = (group['balls_bowled'].shift() / group['balls_per_over']).rolling(n1, min_periods=1).sum()
    group[f'total_overs_throwed_n2'] = (group['balls_bowled'].shift() / group['balls_per_over']).rolling(n2, min_periods=3).sum()
    group[f'total_overs_throwed_n3'] = (group['balls_bowled'].shift() / group['balls_per_over']).rolling(n3, min_periods=5).sum()
    
    group['highest_runs'] = group['runs_scored'].shift().expanding(min_periods=1).max()
    group['highest_wickets'] = group['wickets_taken'].shift().expanding(min_periods=1).max()

    group[f'four_wicket_hauls_n1'] = (group['wickets_taken'] >= 4).shift().rolling(n1, min_periods=1).sum()
    group[f'four_wicket_hauls_n2'] = (group['wickets_taken'] >= 4).shift().rolling(n2, min_periods=3).sum()
    group[f'four_wicket_hauls_n3'] = (group['wickets_taken'] >= 4).shift().rolling(n3, min_periods=5).sum()
   
    return group

def calculate_rolling_fantasy_score(group):
    """Calculate the rolling average of fantasy scores."""
    group['avg_fantasy_score_3'] = group['fantasy_score_total'].shift().rolling(3, min_periods=1).mean()
    group['avg_fantasy_score_5'] = group['fantasy_score_total'].shift().rolling(5, min_periods=2).mean()
    group['avg_fantasy_score_7'] = group['fantasy_score_total'].shift().rolling(7, min_periods=3).mean()
    group['avg_fantasy_score_12'] = group['fantasy_score_total'].shift().rolling(12, min_periods=4).mean()
    group['avg_fantasy_score_15'] = group['fantasy_score_total'].shift().rolling(15, min_periods=5).mean()
    group['avg_fantasy_score_25'] = group['fantasy_score_total'].shift().rolling(25, min_periods=6).mean()

    return group

def calculate_rolling_ducks(group, n1=3,n2=7,n3=12):
    """Calculate the rolling sum of ducks (runs_scored == 0 and player_out == 1) over the last n matches."""
    group['ducks'] = ((group['runs_scored'] == 0) & (group['player_out'] == 1)).astype(int)
    group[f'rolling_ducks_n1'] = group['ducks'].shift().rolling(n1, min_periods=1).sum()
    group[f'rolling_ducks_n2'] = group['ducks'].shift().rolling(n2, min_periods=3).sum()
    group[f'rolling_ducks_n3'] = group['ducks'].shift().rolling(n3, min_periods=5).sum()

    return group

def calculate_rolling_maidens(group, n1=3,n2=7,n3=12):
    """Calculate the rolling sum of maidens over the last n matches."""
    group[f'rolling_maidens_n1'] = group['maidens'].shift().rolling(n1, min_periods=1).sum()
    group[f'rolling_maidens_n2'] = group['maidens'].shift().rolling(n2, min_periods=3).sum()
    group[f'rolling_maidens_n3'] = group['maidens'].shift().rolling(n3, min_periods=5).sum()

    return group


def calculate_alpha_batsmen_score(group, n1=3, n2=7, n3=12):
    """Calculate the α_batsmen_score tailored for Dream11 point prediction in ODIs with multiple time horizons."""
    group = group.sort_values('start_date')

    # Calculate rolling averages for the last n1, n2, and n3 matches
    for i,n in enumerate([n1, n2, n3]):
        group[f'avg_runs_scored_n{i+1}'] = group['runs_scored'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_strike_rate_n{i+1}'] = group[f'strike_rate_n{i+1}'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_sixes_n{i+1}'] = group['sixes_scored'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_fours_n{i+1}'] = group['fours_scored'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_half_centuries_n{i+1}'] = group['half_centuries_cumsum'].shift().rolling(n, min_periods=i+1).sum()
        group[f'avg_centuries_n{i+1}'] = group['centuries_cumsum'].shift().rolling(n, min_periods=i+1).sum()
        group[f'avg_rolling_ducks_n{i+1}'] = group[f'rolling_ducks_n{i+1}'].shift().rolling(n, min_periods=i+1).sum()

    group.fillna(0, inplace=True)

    for i,n in enumerate([n1, n2, n3]):
        group[f'α_batsmen_score_n{i+1}'] = (
        0.25 * group[f'avg_runs_scored_n{i+1}'] +       # Runs scored (core contribution)
        0.20 * group[f'avg_strike_rate_n{i+1}'] +       # Emphasis on strike rate (impact metric)
        0.30 * group[f'avg_half_centuries_n{i+1}'] +    # Rewards for scoring milestones
        0.15 * group[f'avg_sixes_n{i+1}'] +             # Separate bonus for six-hitting
        0.10 * group[f'avg_fours_n{i+1}'] -             # Lower weight for fours
        2.0 * group[f'avg_rolling_ducks_n{i+1}']        # Reduced penalty for ducks
    )

    return group

def calculate_alpha_bowler_score(group, n1=3, n2=7, n3=12):
    """
    Calculate the α_bowler_score tailored for Dream11 point prediction in ODIs 
    with multiple time horizons.
    """
    group = group.sort_values('start_date')

    # Calculate rolling averages for the last n1, n2, and n3 matches
    for i,n in enumerate([n1, n2, n3]):
        group[f'avg_wickets_taken_n{i+1}'] = group['wickets_taken'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_bowling_average_n{i+1}'] = group[f'bowling_average_n{i+1}'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_bowling_strike_rate_n{i+1}'] = group[f'bowling_strike_rate_n{i+1}'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_economy_rate_n{i+1}'] = group[f'economy_rate_n{i+1}'].shift().rolling(n, min_periods=i+1).mean()
        group[f'avg_maidens_n{i+1}'] = group[f'rolling_maidens_n{i+1}'].shift().rolling(n, min_periods=i+1).sum()

    # Replace NaN values with 0 before calculating the α_bowler_score
    group.fillna(0, inplace=True)

    # Calculate the α_bowler_score for each time horizon
    for i,n in enumerate([n1, n2, n3]):
        group[f'α_bowler_score_n{i+1}'] = (
        0.35 * group[f'avg_wickets_taken_n{i+1}'] +           # Wickets taken (core metric for T20 fantasy)
        0.25 * group[f'avg_bowling_strike_rate_n{i+1}'] +     # Strike rate (key for Dream11 in T20s)
        0.20 * group[f'avg_economy_rate_n{i+1}'] +            # Economy rate (penalized in T20s if high)
        0.10 * group[f'avg_maidens_n{i+1}'] -                 # Maidens (rare but valuable in T20)
        0.10 * group[f'avg_bowling_average_n{i+1}']        
        )

    return group

def assign_rating_score(group,n1=3,n2=7,n3=12):
    """
    Assign batsman and bowler ratings based on predefined ranges.
    Parameters:
    - group: DataFrame containing player performance data with 'α_batsmen_score' and 'α_bowler_score'.
    Returns:
    - group: DataFrame with 'batsman_rating' and 'bowler_rating' added.
    """
    batsman_ranges = {
        (0, 5): 0,
        (5, 15): 4,
        (15, 25): 9,
        (25, 35): 16,
        (35, 45): 25,
        (45, 55): 49,
        (55, float('inf')): 81
    }

    bowler_ranges = {
        (0, 1): 0,
        (1, 5): 9,
        (5, 7.5): 16,
        (7.5, 12.5): 25,
        (12.5, 15): 36,
        (15, 17.5): 49,
        (17.5, 20): 64,
        (20, float('inf')): 100
    }

    def get_rating(score, ranges):
        for (lower, upper), rating in ranges.items():
            if lower <= score < upper:
                return rating
        return 0

    group[f'batsman_rating_n1'] = group[f'α_batsmen_score_n1'].apply(lambda x: get_rating(x, batsman_ranges))
    group[f'batsman_rating_n2'] = group[f'α_batsmen_score_n2'].apply(lambda x: get_rating(x, batsman_ranges))
    group[f'batsman_rating_n3'] = group[f'α_batsmen_score_n3'].apply(lambda x: get_rating(x, batsman_ranges))
    
    group[f'bowler_rating_n1'] = group[f'α_bowler_score_n1'].apply(lambda x: get_rating(x, bowler_ranges))
    group[f'bowler_rating_n2'] = group[f'α_bowler_score_n2'].apply(lambda x: get_rating(x, bowler_ranges))
    group[f'bowler_rating_n3'] = group[f'α_bowler_score_n3'].apply(lambda x: get_rating(x, bowler_ranges))
    
    return group

def longtermfeatures(group):
    """Calculate long-term career features for batting and bowling."""
    group = group.sort_values('start_date')

    group['longterm_avg_runs'] = group['runs_scored'].shift().expanding().mean()
    group['longterm_var_runs'] = np.sqrt(group['runs_scored'].shift().expanding().var())
    group['longterm_avg_strike_rate'] = (
        (group['runs_scored'].shift().expanding().sum()) /
        (group['balls_faced'].shift().expanding().sum()) * 100
    )

    group['longterm_avg_wickets_per_match'] = group['wickets_taken'].shift().expanding().mean()
    group['longterm_var_wickets_per_match'] = np.sqrt(group['wickets_taken'].shift().expanding().var())
    group['longterm_avg_economy_rate'] = (
        (group['runs_conceded'].shift().expanding().sum()) /
        ((group['balls_bowled'].shift().expanding().sum()) / group['balls_per_over'].iloc[0])
    )

    return group

def order_seen(group):
    group['order_seen_mode'] = group['order_seen'].shift().expanding().apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
    return group

def year(group):
    group['year'] = group['start_date'].dt.year
    return group

def calculate_30s(runs_scored):
    """Calculate the total number of 30s."""
    return ((runs_scored >= 30) & (runs_scored < 50)).sum()

def run_30_to_50(group):

    group = group.sort_values('start_date')
    group['cumulative_30s'] = group['runs_scored'].shift().expanding(min_periods=1).apply(calculate_30s)
    group['conversion_30_to_50'] = group.apply(lambda x: (x['half_centuries_cumsum'] / x['cumulative_30s']) if x['cumulative_30s'] != 0 else 0, axis=1)
    return group

def preprocess_before_merge(data1,data2):
    """
    Preprocesses the dataframes before merging them.

    Args:
    data1 (pd.DataFrame): The first dataframe to be merged.
    data2 (pd.DataFrame): The second dataframe to be merged.

    Returns:
    pd.DataFrame: Preprocessed data1.
    pd.DataFrame: Preprocessed data2.
    """
    if 'Unnamed: 0' in data1.columns:
        data1.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in data2.columns:
        data2.drop(columns=['Unnamed: 0'], inplace=True)

    data1['match_id'] = data1['match_id'].astype(str)
    data2['match_id'] = data2['match_id'].astype(str)

    data1['player_id'] = data1['player_id'].astype(str)
    data2['player_id'] = data2['player_id'].astype(str)

    data1['start_date'] = pd.to_datetime(data1['start_date'])
    data2['start_date'] = pd.to_datetime(data2['start_date'])

    data1 = data1.sort_values(by='start_date').reset_index(drop=True)
    data2 = data2.sort_values(by='start_date').reset_index(drop=True)
    print("Preprocessing before merging completed for both dataframes.")
    return data1,data2

def get_hemisphere(country):
    southern_hemisphere = [
        'New Zealand', 'Australia', 'South Africa', 'Argentina', 'Chile', 
        'Uruguay', 'Zimbabwe', 'Namibia', 'Botswana', 'Fiji', 'Malawi', 
        'Papua New Guinea', 'Samoa'
    ]
    
    if country in southern_hemisphere:
        return 'southern'
    
    northern_hemisphere = [
        'Barbados', 'United States', 'United Kingdom', 'Sri Lanka', 'Canada', 
        'India', 'Pakistan', 'Bangladesh', 'United Arab Emirates', 'Kenya', 
        'Malaysia', 'Japan', 'Netherlands', 'Sweden', 'Hong Kong', 'Thailand', 
        'Nepal', 'Uganda', 'Trinidad and Tobago', 'Ireland', 'Portugal', 
        'Kuwait', 'Italy', 'Singapore', 'Korea, Republic of', 'Philippines', 
        'Saint Kitts and Nevis', 'Czechia', 'Bulgaria', 'Germany', 'Finland', 
        'Morocco', 'Qatar', 'Cambodia', 'Gibraltar', 'Denmark', 'Dominica', 
        'China', 'Cameroon', 'Ghana', 'France', 
        'Saint Vincent and the Grenadines', 'Croatia', 'Norway', 'Serbia', 
        'Greece'
    ]
    
    if country in northern_hemisphere:
        return 'northern'
    
    if pd.isna(country) or country == 'Unknown Country':
        return 'Unknown'
    
    return 'northern'

def get_season(date, hemisphere):
    month = date.month
    if hemisphere == 'northern':
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
    elif hemisphere == 'southern':
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        elif month in [9, 10, 11]:
            return 'Spring'
    return 'Unknown'

def get_test_data(df):
    df=df[(df['match_type']=='MDM') | (df['match_type']=='Test')]
    return df

def get_T20_data(df):
    df=df[(df['match_type']=='T20') | (df['match_type']=='IT20')]
    return df

def get_ODI_data(df):
    df=df[(df['match_type']=='ODI') | (df['match_type']=='ODM')]
    return df

class FeatureGeneration:
    def __init__(self, mw_overall, mw_pw_profile,match_type):
        """
        Initializes the class with the dataframes required for processing.
        
        Args:
            mw_overall (pd.DataFrame): The first dataframe containing country and home/away data.
            mw_pw_profile (pd.DataFrame): The second dataframe containing match or player data.
        """
        self.mw_overall = mw_overall
        self.mw_pw_profile = mw_pw_profile
        self._is_preprocessed = False


        self.match_type = match_type

        self.HelperFunctions = {
            "ODI": {'match_type_data':get_ODI_data,'calculate_rolling_batting_stats':calculate_rolling_batting_stats_test,'calculate_rolling_bowling_stats':calculate_rolling_bowling_stats_test},
            "T20": {'match_type_data':get_T20_data,'calculate_rolling_batting_stats':calculate_rolling_batting_stats_test,'calculate_rolling_bowling_stats':calculate_rolling_bowling_stats_test},
            "Test": {'match_type_data':get_test_data,'calculate_rolling_batting_stats':calculate_rolling_batting_stats_test,'calculate_rolling_bowling_stats':calculate_rolling_bowling_stats_test},
        }

    
    def get_match_type_data(self):
        self.mw_pw_profile=self.HelperFunctions[self.match_type]['match_type_data'](self.mw_pw_profile)
    
    def _preprocess(self):
        if 'Unnamed: 0' in self.mw_pw_profile.columns:
            self.mw_pw_profile.drop(columns=['Unnamed: 0'], inplace=True)
        if 'Unnamed: 0' in self.mw_overall.columns:
            self.mw_overall.drop(columns=['Unnamed: 0'], inplace=True)

        self.mw_overall['match_id'] = self.mw_overall['match_id'].astype(str)
        self.mw_pw_profile['match_id'] = self.mw_pw_profile['match_id'].astype(str)

        self.mw_pw_profile['player_id'] = self.mw_pw_profile['player_id'].astype(str)

        self.mw_pw_profile['start_date'] = pd.to_datetime(self.mw_pw_profile['start_date'])
        self.mw_pw_profile = self.mw_pw_profile.sort_values(by='start_date').reset_index(drop=True)
        self._is_preprocessed = True
        print("Preprocessing completed for both dataframes.")

    def _ensure_preprocessed(self):
        """
        Ensures preprocessing is done before any method is executed.
        """
        if not self._is_preprocessed:
            self._preprocess()
    
    def drop_columns(self,columns_to_drop):
        """
        Drops the specified columns from the mw_pw_profile dataframe.

        Args:
        columns_to_drop (list): List of columns to drop.

        Returns:
        pd.DataFrame: Updated mw_pw_profile with the specified columns dropped.
        """
        self._ensure_preprocessed()
        self.mw_pw_profile.drop(columns=columns_to_drop,inplace=True)


    def process_country_and_homeaway(self):
        """
        Processes the data to map city names to country names and determine home/away status.

        Returns:
        pd.DataFrame: Updated mw_pw_profile with 'country_ground' and 'home_away' columns.
        """
        self._ensure_preprocessed()

        gc = geonamescache.GeonamesCache()
        cities = gc.get_cities()
        city_to_countrycode = {info['name']: info['countrycode'] for code, info in cities.items()}

        def get_country(city_name):
            country_code = city_to_countrycode.get(city_name)
            if not country_code:
                return "Unknown Country"
            country = pycountry.countries.get(alpha_2=country_code)
            return country.name if country else "Unknown Country"

        self.mw_overall['country_ground'] = self.mw_overall['city'].apply(get_country)

        match_id_to_country = self.mw_overall.set_index('match_id')['country_ground'].to_dict()

        self.mw_pw_profile['country_ground'] = self.mw_pw_profile['match_id'].map(match_id_to_country)

        def homeaway(country_of_player, country_venue):
            if country_venue == 'Unknown Country':
                return 'neutral'
            elif country_of_player == country_venue:
                return 'home'
            else:
                return 'away'

        self.mw_pw_profile['home_away'] = self.mw_pw_profile.apply(
            lambda row: homeaway(row['player_team'], row['country_ground']), axis=1
        )

    def calculate_fantasy_scores(self):
        """
        Calculates fantasy scores for batting and bowling based on the match data.

        Returns:
        pd.DataFrame: Updated mw_pw_profile with fantasy scores.
        """
        self._ensure_preprocessed()
        
        df = self.mw_pw_profile
        df['fantasy_score_batting'] = 0
        df['fantasy_score_bowling'] = 0
        df['fantasy_score_total'] = 0
        
        for index, row in df.iterrows():
            # Batting fantasy score calculation
            runs_scored = row['runs_scored']
            balls_faced = row['balls_faced']
            fours_scored = row['fours_scored']
            sixes_scored = row['sixes_scored']
            catches_taken = row['catches_taken']
            match_type = row['match_type']
            series = row['series_name']
            player_out = row['player_out']
            stumpings = row['stumpings_done']
            fantasy_playing = 0
            fantasy_batting = 0
            
            if series == 'T10':
                fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 50:
                    fantasy_batting += 16
                elif runs_scored >= 30:
                    fantasy_batting += 8

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 2

                if balls_faced >= 5:
                    strike_rate = (runs_scored / balls_faced) * 100
                    if strike_rate < 60:
                        fantasy_batting -= 6
                    elif strike_rate < 70:
                        fantasy_batting -= 4
                    elif strike_rate <= 80:
                        fantasy_batting -= 2
                    elif strike_rate > 190:
                        fantasy_batting += 6
                    elif strike_rate > 170:
                        fantasy_batting += 4
                    elif strike_rate >= 150:
                        fantasy_batting += 2

            elif series == '6ixty':
                fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 50:
                    fantasy_batting += 16
                elif runs_scored >= 30:
                    fantasy_batting += 8

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 2

                if balls_faced >= 5:
                    strike_rate = (runs_scored / balls_faced) * 100
                    if strike_rate < 60:
                        fantasy_batting -= 6
                    elif strike_rate < 70:
                        fantasy_batting -= 4
                    elif strike_rate <= 80:
                        fantasy_batting -= 2
                    elif strike_rate > 190:
                        fantasy_batting += 6
                    elif strike_rate > 170:
                        fantasy_batting += 4
                    elif strike_rate >= 150:
                        fantasy_batting += 2

            elif series == 'The100':
                fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 100:
                    fantasy_batting += 20
                elif runs_scored >= 50:
                    fantasy_batting += 10
                elif runs_scored >= 30:
                    fantasy_batting += 5

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 2

            elif match_type in ['T20', 'IT20']:
                if match_type == 'T20':
                    fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 100:
                    fantasy_batting += 16
                elif runs_scored >= 50:
                    fantasy_batting += 8
                elif runs_scored >= 30:
                    fantasy_batting += 4

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 2

                if balls_faced >= 10:
                    strike_rate = (runs_scored / balls_faced) * 100
                    if strike_rate < 50:
                        fantasy_batting -= 6
                    elif strike_rate < 60:
                        fantasy_batting -= 4
                    elif strike_rate <= 70:
                        fantasy_batting -= 2
                    elif strike_rate > 170:
                        fantasy_batting += 6
                    elif strike_rate > 150:
                        fantasy_batting += 4
                    elif strike_rate >= 130:
                        fantasy_batting += 2

            elif match_type in ['ODI', 'ODM']:
                if match_type == 'ODI':
                    fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 100:
                    fantasy_batting += 8
                elif runs_scored >= 50:
                    fantasy_batting += 4

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 3

                if balls_faced >= 20:
                    strike_rate = (runs_scored / balls_faced) * 100
                    if strike_rate < 30:
                        fantasy_batting -= 6
                    elif strike_rate < 40:
                        fantasy_batting -= 4
                    elif strike_rate <= 50:
                        fantasy_batting -= 2
                    elif strike_rate > 140:
                        fantasy_batting += 6
                    elif strike_rate > 120:
                        fantasy_batting += 4
                    elif strike_rate >= 100:
                        fantasy_batting += 2

            elif match_type in ['Test', 'MDM']:
                if match_type == 'Test':
                    fantasy_playing = 4
                fantasy_batting = 1 * runs_scored + 1 * (fours_scored) + 2 * (sixes_scored) + 12 * stumpings
                if runs_scored >= 100:
                    fantasy_batting += 8
                elif runs_scored >= 50:
                    fantasy_batting += 4

                if runs_scored == 0 and player_out:
                    fantasy_batting -= 4

            df.at[index, 'fantasy_score_batting'] = fantasy_batting

            # Bowling fantasy score calculation
            balls_bowled = row['balls_bowled']
            runs_conceded = row['runs_conceded']
            wickets_taken = row['wickets_taken']
            maidens = row['maidens']
            bowled_done = row['bowled_done']
            lbw_done = row['lbw_done']
            run_out_direct = row['run_out_direct']
            run_out_throw = row['run_out_throw']

            fantasy_bowling = 0
            if series == 'T10':
                fantasy_bowling = 25 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 16 * maidens + 8 * catches_taken
                if wickets_taken >= 3:
                    fantasy_bowling += 16
                elif wickets_taken >= 2:
                    fantasy_bowling += 8

                if balls_bowled >= 6:
                    economy_rate = (runs_conceded / balls_bowled) * 6
                    if economy_rate < 7:
                        fantasy_bowling += 6
                    elif economy_rate < 8:
                        fantasy_bowling += 4
                    elif economy_rate <= 9:
                        fantasy_bowling += 2
                    elif economy_rate > 16:
                        fantasy_bowling -= 6
                    elif economy_rate > 15:
                        fantasy_bowling -= 4
                    elif economy_rate >= 14:
                        fantasy_bowling -= 2
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

                if catches_taken >= 3:
                    fantasy_bowling += 4

            elif series == '6ixty':
                fantasy_bowling = 25 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 16 * maidens + 8 * catches_taken
                if wickets_taken >= 3:
                    fantasy_bowling += 16
                elif wickets_taken >= 2:
                    fantasy_bowling += 8

                if balls_bowled >= 6:
                    economy_rate = (runs_conceded / balls_bowled) * 6
                    if economy_rate < 7:
                        fantasy_bowling += 6
                    elif economy_rate < 8:
                        fantasy_bowling += 4
                    elif economy_rate <= 9:
                        fantasy_bowling += 2
                    elif economy_rate > 16:
                        fantasy_bowling -= 6
                    elif economy_rate > 15:
                        fantasy_bowling -= 4
                    elif economy_rate >= 14:
                        fantasy_bowling -= 2
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

                if catches_taken >= 3:
                    fantasy_bowling += 4

            elif series == 'The100':
                fantasy_bowling = 25 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 8 * catches_taken
                if wickets_taken >= 5:
                    fantasy_bowling += 20
                elif wickets_taken >= 4:
                    fantasy_bowling += 10
                elif wickets_taken >= 3:
                    fantasy_bowling += 5
                elif wickets_taken >= 2:
                    fantasy_bowling += 3
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

                if catches_taken >= 3:
                    fantasy_bowling += 4

            elif match_type in ['T20', 'IT20']:
                fantasy_bowling = 25 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 12 * maidens + 8 * catches_taken
                if wickets_taken >= 5:
                    fantasy_bowling += 16
                elif wickets_taken >= 4:
                    fantasy_bowling += 8
                elif wickets_taken >= 3:
                    fantasy_bowling += 4

                if catches_taken >= 3:
                    fantasy_bowling += 4

                if balls_bowled >= 12:
                    economy_rate = (runs_conceded / balls_bowled) * 6
                    if economy_rate < 5:
                        fantasy_bowling += 6
                    elif economy_rate < 6:
                        fantasy_bowling += 4
                    elif economy_rate <= 7:
                        fantasy_bowling += 2
                    elif economy_rate > 12:
                        fantasy_bowling -= 6
                    elif economy_rate > 11:
                        fantasy_bowling -= 4
                    elif economy_rate >= 10:
                        fantasy_bowling -= 2
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

            elif match_type in ['ODI', 'ODM']:
                fantasy_bowling = 25 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 4 * maidens + 8 * catches_taken
                if wickets_taken >= 5:
                    fantasy_bowling += 8
                elif wickets_taken >= 4:
                    fantasy_bowling += 4

                if catches_taken >= 3:
                    fantasy_bowling += 4

                if balls_bowled >= 30:
                    economy_rate = (runs_conceded / balls_bowled) * 6
                    if economy_rate < 2.5:
                        fantasy_bowling += 6
                    elif economy_rate < 3.5:
                        fantasy_bowling += 4
                    elif economy_rate <= 4.5:
                        fantasy_bowling += 2
                    elif economy_rate > 9:
                        fantasy_bowling -= 6
                    elif economy_rate > 8:
                        fantasy_bowling -= 4
                    elif economy_rate >= 7:
                        fantasy_bowling -= 2
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

            elif match_type in ['Test', 'MDM']:
                fantasy_bowling = 16 * (wickets_taken) + 8 * (lbw_done + bowled_done) + 8 * catches_taken
                if wickets_taken >= 5:
                    fantasy_bowling += 8
                elif wickets_taken >= 4:
                    fantasy_bowling += 4
                fantasy_bowling += run_out_throw * 6 + run_out_direct * 12

            df.at[index, 'fantasy_score_bowling'] = fantasy_bowling
            df.at[index, 'fantasy_score_total'] += fantasy_playing

        df['fantasy_score_total'] += df['fantasy_score_batting'] + df['fantasy_score_bowling']
        self.mw_pw_profile = df

    def player_features(self):
        """Main function to calculate player features using helper functions."""

        feature_data = []
        df = self.mw_pw_profile
        for (name, match_type), group in df.groupby(['player_id', 'match_type']):
            group = group.sort_values('start_date')
            group= longtermfeatures(group)
            group = self.HelperFunctions[self.match_type]['calculate_rolling_batting_stats'](group)
            group = self.HelperFunctions[self.match_type]['calculate_rolling_bowling_stats'](group)
            group = calculate_centuries_and_half_centuries(group)
            group = calculate_additional_stats(group)
            group = calculate_rolling_fantasy_score(group)
            group = calculate_rolling_ducks(group)
            group = calculate_rolling_maidens(group)
            group = calculate_alpha_batsmen_score(group)
            group = calculate_alpha_bowler_score(group)
            group=assign_rating_score(group)
            group=order_seen(group)
            group=year(group)
            group=run_30_to_50(group)

            feature_data.append(group[['player_id','match_id' ,'match_type', 'start_date', 
                            f'batting_average_n1', f'strike_rate_n1', f'boundary_percentage_n1',
                            f'batting_average_n2', f'strike_rate_n2', f'boundary_percentage_n2',
                            f'batting_average_n3', f'strike_rate_n3', f'boundary_percentage_n3',
                            'centuries_cumsum', 'half_centuries_cumsum', 
                            f'economy_rate_n1', f'economy_rate_n2', f'economy_rate_n3',
                            f'wickets_in_n1_matches', f'wickets_in_n2_matches', f'wickets_in_n3_matches',
                            f'total_overs_throwed_n1', f'total_overs_throwed_n2', f'total_overs_throwed_n3',
                            f'bowling_average_n1', f'bowling_strike_rate_n1', f'bowling_average_n2', f'bowling_strike_rate_n2',
                            f'bowling_average_n3', f'bowling_strike_rate_n3',f'CBR', f'fielding_points',
                           
                            f'four_wicket_hauls_n1', f'four_wicket_hauls_n2', f'four_wicket_hauls_n3',
                            'highest_runs', 'highest_wickets', 'longterm_avg_runs', 'longterm_var_runs',
                            'order_seen_mode', f'rolling_ducks_n1', f'rolling_maidens_n1', 
                            f'rolling_ducks_n2', f'rolling_maidens_n2', f'rolling_ducks_n3', f'rolling_maidens_n3',
                            'longterm_avg_strike_rate', 'longterm_avg_wickets_per_match', 'longterm_var_wickets_per_match', 
                            'longterm_avg_economy_rate', 'avg_fantasy_score_3', 'avg_fantasy_score_5','avg_fantasy_score_7', 
                            'avg_fantasy_score_12', 'avg_fantasy_score_15','avg_fantasy_score_25',
                            f'batsman_rating_n1', f'bowler_rating_n1',f'batsman_rating_n2', f'bowler_rating_n2',
                            f'batsman_rating_n3', f'bowler_rating_n3',
                            f'α_batsmen_score_n1', f'α_batsmen_score_n2', f'α_batsmen_score_n3', 
                            f'α_bowler_score_n1', f'α_bowler_score_n2', f'α_bowler_score_n3',
                            'year','conversion_30_to_50']])

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)  # Reset index and drop old index

        self.mw_pw_profile=df.merge(result_df,on=['player_id','match_id','match_type','start_date'],how='left')
    
    def avg_of_player_against_opposition(self):
        """
        Calculate the average of a player against a particular opposition.
        """
        self._ensure_preprocessed()
        final_df = self.mw_pw_profile
        final_df['avg_against_opposition'] = (
            final_df.groupby(['player_id', 'opposition_team'])['fantasy_score_total']
            .apply(lambda x: x.shift().expanding().mean())
            .reset_index(level=[0, 1], drop=True)  
        )
        self.mw_pw_profile=final_df

    def add_season(self):
        """
        Add season feature to the dataframe.
        """
        self._ensure_preprocessed()
        final_df = self.mw_pw_profile
        final_df['hemisphere'] = final_df['country_ground'].apply(get_hemisphere)
        final_df['season'] = final_df.apply(
            lambda row: get_season(row['start_date'], row['hemisphere']) 
            if row['hemisphere'] != 'Unknown' else 'Unknown', axis=1
        )
        self.mw_pw_profile=final_df
        self.drop_columns(['hemisphere'])

    def batter_bowler_classification(self):
        final_df = self.mw_pw_profile
        final_df['batter'] = (final_df['playing_role'].str.contains('Batter|Allrounder|WicketKeeper|None', na=False) | final_df['playing_role'].isnull()).astype(int)
        final_df['bowler'] = (final_df['playing_role'].str.contains('Bowler|Allrounder|None', na=False) | final_df['playing_role'].isnull()).astype(int)
        self.mw_pw_profile=final_df

    def categorize_bowling_style(self):
        def define_styles(style):
            style = str(style)  # Convert style to string
            if pd.isna(style) or style == "None":
                return "Others"
            elif "Right arm Fast" in style or "Right arm Medium fast" in style:
                return "Fast"
            elif "Right arm Offbreak" in style or "Legbreak" in style or "Googly" in style:
                return "Spin"
            elif "Slow Left arm Orthodox" in style or "Left arm Wrist spin" in style or "Left arm Slow" in style:
                return "Spin"
            elif "Left arm Fast" in style or "Left arm Medium" in style:
                return "Fast"
            else:
                if "Medium" in style or "Slow" in style:
                    return "Medium"
                else:
                    return "Others"
        self.mw_pw_profile['bowling_style'] = self.mw_pw_profile['bowling_style'].apply(define_styles)

    def sena_sub_countries(self):
        """
        Creates new columns based on the player's role and match location for Test matches.
        This includes categorizing players as batsmen and bowlers from subcontinent and SENA countries.
        If country_ground is missing, sets all columns to 0.
        """
        df = self.mw_pw_profile
        subcontinent_countries = ['India', 'Pakistan', 'Sri Lanka', 'Bangladesh', 'Nepal', 'Afghanistan', 'Zimbabwe', 'Bhutan', 'Maldives']

        sena_countries = ['South Africa', 'England', 'Australia', 'New Zealand', 'West Indies', 'Ireland', 'Scotland']
        
        # Initialize new columns with 0 values
        df['batsman_sena_sub'] = 0
        df['batsman_sub_sena'] = 0
        df['bowler_sub_sena'] = 0
        df['bowler_sena_sub'] = 0
        df['batsman_sena_sena'] = 0
        df['bowler_sena_sena'] = 0
        df['bowler_sub_sub'] = 0
        df['batsman_sub_sub'] = 0

        bowler = df[df['bowler'] == 1]
        batsman = df[df['batter'] == 1]
        neither = df[(df['batter'] == 0) & (df['bowler'] == 0)]
        for idx, row in batsman.iterrows():
            # Extract data for each player
        
        
            player_team = row['player_team']
            match_location = row['country_ground']  # Use country_ground for the match location

            # If match_location is missing (NaN), set all columns to 0 and skip further processing
            if pd.isna(match_location):
                continue

            else:
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'batsman_sub_sub'] = 1  # Batsman from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'batsman_sub_sena'] = -1  # Batsman from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'batsman_sena_sub'] = -1  # Batsman from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'batsman_sena_sena'] = 1  # Batsman from SENA playing in SENA
            

        for idx, row in bowler.iterrows():
            bowling_style = row['bowling_style']
            player_team = row['player_team']
            match_location = row['country_ground']
            if pd.isna(match_location):
                continue

            if bowling_style == 'Spin':  # Spinner
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sub_sub'] = 1  # Spinner from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sub_sena'] = -1  # Spinner from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sena_sub'] = -1 # Spinner from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sena_sena'] = 1 # Spinner from SENA playing in SENA

            elif bowling_style == 'Fast':  # Pace Bowler
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sub_sub'] = 0  # Pace bowler from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sub_sena'] = 1  # Pace bowler from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sena_sub'] = -1  # Pace bowler from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sena_sena'] = 1 # Pace bowler from SENA playing in SENA
        for idx, row in neither.iterrows():
            bowling_style = row['bowling_style']
            player_team = row['player_team']
            match_location = row['country_ground']   
            if pd.isna(match_location):
                continue

            else:
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'batsman_sub_sub'] = 1  # Batsman from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'batsman_sub_sena'] = -1  # Batsman from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'batsman_sena_sub'] = -1  # Batsman from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'batsman_sena_sena'] = 1  # Batsman from SENA playing in SENA  

            if bowling_style == 'Spin':  # Spinner
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sub_sub'] = 1  # Spinner from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sub_sena'] = -1  # Spinner from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sena_sub'] = -1 # Spinner from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sena_sena'] = 1 # Spinner from SENA playing in SENA

            elif bowling_style == 'Fast':  # Pace Bowler
                if player_team in subcontinent_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sub_sub'] = 0  # Pace bowler from subcontinent playing in subcontinent
                elif player_team in subcontinent_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sub_sena'] = 1  # Pace bowler from subcontinent playing in SENA
                elif player_team in sena_countries and match_location in subcontinent_countries:
                    df.at[idx, 'bowler_sena_sub'] = -1  # Pace bowler from SENA playing in subcontinent
                elif player_team in sena_countries and match_location in sena_countries:
                    df.at[idx, 'bowler_sena_sena'] = 1 # Pace bowler from SENA playing in SENA
            

        self.mw_pw_profile=df

    def calculate_match_level_venue_stats(self, lower_param=4.5,upper_param=7):
        # Ensure only the first date is used if dates contain multiple entries
        df = self.mw_overall
        df['dates'] = df['dates'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
        
        # Convert dates column to datetime format for proper sorting
        df['dates'] = pd.to_datetime(df['dates'])

        # Ensure the dataframe is sorted by venue, match_type, and dates
        df = df.sort_values(['venue', 'match_type', 'dates'])

        # Replace NaNs in dismissal columns with 0 (since they represent counts of dismissals)
        dismissal_cols = ['bowled', 'caught', 'lbw', 'caught and bowled', 
                        'run out', 'stumped', 'hit wicket', 'retired hurt', 
                        'retired not out', 'obstructing the field', 
                        'retired out', 'handled the ball', 'hit the ball twice']
        df[dismissal_cols] = df[dismissal_cols].fillna(0)

        # Rolling calculation function
        def rolling_stats(group):
            balls_per_inning = 120 if group['match_type'].iloc[0] == 'T20' else 300

        # Calculate the row number within each group to determine the number of innings processed
            group = group.reset_index(drop=True)  # Ensure a contiguous index within the group
            group['inning_number'] = group.index

        # Calculate cumulative stats up to (but excluding) the current match
            group['cumulative_runs'] = group['runs_off_bat'].shift(1).fillna(0).cumsum()
            group['cumulative_wickets'] = (group[dismissal_cols].sum(axis=1).shift(1).fillna(0).cumsum())

        # Cumulative balls based on the inning number
            group['cumulative_balls'] = group['inning_number'] * balls_per_inning
            group['overs'] = group['cumulative_balls'] / 6  # Convert balls to overs

        # Calculate ARPO
            group['ARPO_venue'] = group['cumulative_runs'] / group['overs']

        # Derived stats
            group['Boundary_Percentage_venue'] = (group['cumulative_runs'] / group['cumulative_balls']) * 100
            group['BSR_venue'] = (group['cumulative_runs'] / group['cumulative_balls']) * 100

        # Average First-Innings Score (AFIS)
            group['AFIS_venue'] = group[group['innings'] == 1]['runs_off_bat'].expanding().mean()

        # Classify pitch type based on ARPO and two thresholds
            group['Pitch_Type'] = group['ARPO_venue'].apply(
                lambda x: 'Bowling-Friendly' if x < lower_param else 
                    'Batting-Friendly' if x > upper_param else 
                    'Neutral'
        )

            return group



        # Group by venue and match_type, then apply rolling stats
        df = df.groupby(['venue', 'match_type']).apply(rolling_stats)

        # Consolidate by match_id for final output
        match_stats = df.groupby('match_id').agg({
            'venue': 'first',
            'match_type': 'first',
            'dates': 'first',
            'ARPO_venue': 'last',                     # ARPO as of this match
            'Boundary_Percentage_venue': 'last',      # Boundary Percentage
            'BSR_venue': 'last',                      # Batting Strike Rate
            'AFIS_venue': 'last',                     # Average First-Innings Score
            'Pitch_Type': 'last'                # Pitch classification
        }).reset_index()

        match_stats.drop(columns=['dates','venue','match_type'], inplace=True)
        self.mw_pw_profile = self.mw_pw_profile.merge(match_stats, on='match_id', how='left')

    def calculate_matches_played_before(self):
        df = self.mw_pw_profile

        df = df.sort_values(by=['player_id', 'match_type', 'start_date'])
        df['longterm_total_matches_of_type'] = df.groupby(['player_id', 'match_type']).cumcount()
        
        self.mw_pw_profile = df

    def calculate_rolling_fantasy_scores_batter_and_bowler(self):
        """
        Adds rolling average fantasy scores for the last 5 matches for bowlers and batters 
        to the input DataFrame based on player roles.

        Args:
            final_df (pd.DataFrame): Input DataFrame with columns 'player_id', 'match_type', 
                                    'start_date', 'fantasy_score_batting', 'fantasy_score_bowling',
                                    'bowler', and 'batter'.

        Returns:
            pd.DataFrame: Updated DataFrame with 'avg_bowler_fantasy_score_5' and 
                        'avg_batter_fantasy_score_5' columns added.


        """
        final_df = self.mw_pw_profile

        feature_data = []

        # Grouping by player_id and match_type
        for (player_id, match_type), group in final_df.groupby(['player_id', 'match_type']):
            group = group.sort_values('start_date')  # Ensure chronological order

            # Case 1: Bowler only
            bowler_only = group[(group['bowler'] == 1) & (group['batter'] == 0)]
            bowler_only['avg_bowler_fantasy_score_5'] = (
                bowler_only['fantasy_score_bowling']
                .shift()
                .rolling(5, min_periods=1)
                .mean()
            )
            bowler_only['avg_batter_fantasy_score_5'] = None

            # Case 2: Batter only
            batter_only = group[(group['bowler'] == 0) & (group['batter'] == 1)]
            batter_only['avg_batter_fantasy_score_5'] = (
                batter_only['fantasy_score_batting']
                .shift()
                .rolling(5, min_periods=1)
                .mean()
            )
            batter_only['avg_bowler_fantasy_score_5'] = None

            # Case 3: All-rounder (both bowler and batter)
            all_rounder = group[(group['bowler'] == 1) & (group['batter'] == 1)]
            all_rounder['avg_bowler_fantasy_score_5'] = (
                all_rounder['fantasy_score_bowling']
                .shift()
                .rolling(5, min_periods=1)
                .mean()
            )
            all_rounder['avg_batter_fantasy_score_5'] = (
                all_rounder['fantasy_score_batting']
                .shift()
                .rolling(5, min_periods=1)
                .mean()
            )

            # Case 4: Neither bowler nor batter
            neither = group[(group['bowler'] == 0) & (group['batter'] == 0)]
            neither['avg_bowler_fantasy_score_5'] = None
            neither['avg_batter_fantasy_score_5'] = None

            # Combine all cases back together
            combined_group = pd.concat([bowler_only, batter_only, all_rounder, neither])
            feature_data.append(combined_group)

        # Combine all groups back into a single DataFrame
        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)
        result_df['avg_bowler_fantasy_score_5'] = result_df['avg_bowler_fantasy_score_5'].fillna(0)
        result_df['avg_batter_fantasy_score_5'] = result_df['avg_batter_fantasy_score_5'].fillna(0)

        self.mw_pw_profile = result_df

    def avg_of_opponent(self):
        final_df = self.mw_pw_profile
        feature_data = []

        # Grouping by player_id, match_type, and opposition_team
        for (player_id, match_type, opposition_team), group in final_df.groupby(['player_id', 'match_type', 'opposition_team']):
            group = group.sort_values('start_date')  # Ensure chronological order
            
            # Determine role of player
            bowler_only = (group['bowler'].iloc[0] == 1) & (group['batter'].iloc[0] == 0)
            batter_only = (group['bowler'].iloc[0] == 0) & (group['batter'].iloc[0] == 1)
            all_rounder = (group['bowler'].iloc[0] == 1) & (group['batter'].iloc[0] == 1)
            neither = (group['bowler'].iloc[0] == 0) & (group['batter'].iloc[0] == 0)

            # Filter opponents' data
            opponent_group = final_df[(final_df['player_team'] == opposition_team) & 
                                    (final_df['match_type'] == match_type) 
                                    ]
            
            # Calculate average opponent scores based on role
            if bowler_only:
                avg_opponent_score = opponent_group[
                    ((opponent_group['bowler'] == 1) & (opponent_group['batter'] == 1)) | 
                    ((opponent_group['bowler'] == 0) & (opponent_group['batter'] == 1))
                ]['avg_batter_fantasy_score_5'].mean()
            elif batter_only:
                filtered_group = opponent_group[
                    ((opponent_group['bowler'] == 1) & (opponent_group['batter'] == 1)) | 
                    ((opponent_group['bowler'] == 1) & (opponent_group['batter'] == 0))
                ]

                if not filtered_group.empty:
                    avg_opponent_score = filtered_group['avg_batter_fantasy_score_5'].mean()
                else:
                    avg_opponent_score = None # No opponent data available
                # print(avg_opponent_score)
            elif all_rounder or neither:
                avg_batter_score = opponent_group['avg_batter_fantasy_score_5'].mean()
                avg_bowler_score = opponent_group['avg_bowler_fantasy_score_5'].mean()
                avg_opponent_score = (avg_batter_score + avg_bowler_score) / 2
            else:
                avg_opponent_score = None

            # Assign calculated score to group
            group = group.copy()  # Avoid SettingWithCopyWarning
            group['avg_of_opponent'] = avg_opponent_score
            feature_data.append(group)

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)

        self.mw_pw_profile = result_df


    def player_features_dot_balls(self):
        data = self.mw_pw_profile
        feature_data = []

        for (name, match_type), group in data.groupby(['player_id', 'match_type']):
            group = group.sort_values('start_date')
            group = rolling_dot_balls_features(group)
            group = longtermfeatures_dot_balls(group)
            feature_data.append(group[['player_id','match_id' ,'match_type', 'start_date',f'dot_ball_percentage_n1',
                                    f'dot_ball_percentage_n2'
                                    ,f'dot_ball_percentage_n3','longterm_dot_ball_percentage',
                                    'longterm_var_dot_ball_percentage']]) 

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)  # Reset index and drop old index

        self.mw_pw_profile = data.merge(result_df,on=['player_id','match_id','match_type','start_date'],how='inner')

    def get_role_factor(self):
        def check_order(position):
            if position <= 3:  # Top Order
                return 1.2
            elif position <= 6:  # Middle Order
                return 1.0
            else:  # Lower Order
                return 0.8

        self.mw_pw_profile['role_factor'] = self.mw_pw_profile['order_seen_mode'].apply(check_order)

    def encode_preprocess(self):
        def one_hot_encode(X, column_name):
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

        def target_encode(df, column, target, smoothing=1):
            """
            Perform target encoding on a categorical column.

            Parameters:
            - df (pd.DataFrame): The input DataFrame.
            - column (str): The column to encode.
            - target (str): The target variable for encoding.
            - smoothing (float): Smoothing factor to balance the global mean and group-specific mean. Higher values give more weight to the global mean.

            Returns:
            - pd.Series: A series containing the target-encoded values.
            """
            global_mean = df[target].mean()
            
            agg = df.groupby(column)[target].agg(['mean', 'count'])
            
            # Compute the smoothed target mean
            smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - 1) / smoothing))
            agg['smoothed_mean'] = global_mean * (1 - smoothing_factor) + agg['mean'] * smoothing_factor
            
            encoded_series = df[column].map(agg['smoothed_mean'])
            
            return encoded_series

        def encode_playing_role_vectorized(df, column='playing_role'):
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

            # Handle non-null playing_role
            non_null_roles = df[column].fillna("None").str.lower()  # Convert to lowercase for consistency

            # Vectorized checks for roles
            df['batter'] += non_null_roles.str.contains("batter").astype(int)
            df['wicketkeeper'] += non_null_roles.str.contains("wicketkeeper").astype(int)
            df['bowler'] += non_null_roles.str.contains("bowler").astype(int)
            df['allrounder'] += non_null_roles.str.contains("allrounder").astype(int)

            # Handle cases where "Allrounder" specifies "Batting" or "Bowling"
            df['batter'] += non_null_roles.str.contains("allrounder.*batting").astype(int)
            df['bowler'] += non_null_roles.str.contains("allrounder.*bowling").astype(int)

            return df[['batter', 'wicketkeeper', 'bowler', 'allrounder']]
        
        
        final_df = self.mw_pw_profile
        final_df['batting_style'].fillna('Right hand Bat', inplace=True)
        final_df = one_hot_encode(final_df, 'match_type')
        final_df = one_hot_encode(final_df, 'batting_style')
        final_df = one_hot_encode(final_df,'gender')
        final_df = one_hot_encode(final_df,'home_away')
        final_df = one_hot_encode(final_df,'season')

        final_df.drop(columns=['bowler','batter'], inplace=True)

        final_df['bowling_style']= target_encode(final_df,'bowling_style','fantasy_score_total')

        final_df[['batter', 'wicketkeeper', 'bowler', 'allrounder']] = encode_playing_role_vectorized(final_df)

        self.mw_pw_profile = final_df
      
    def calculate_rolling_gini_and_caa_with_original_data(self):
        """
        Calculate Gini coefficient and Consistency Adjusted Average (CAA) for each player 
        in a rolling manner and return the original data with added columns.

        Parameters:
        - data (DataFrame): Player dataset with scores and dates.
        - score_column (str): Column name for player scores.
        - date_column (str): Column name for match dates.

        Returns:
        - data_with_metrics (DataFrame): Original DataFrame with added Gini coefficient 
        and CAA columns.
        """
        data = self.mw_pw_profile

        score_column, date_column='runs_scored', 'start_date'
        def gini_and_caa(scores):
            """Helper function to calculate Gini coefficient and CAA."""
            scores = np.array(scores)
            n = len(scores)
            if n == 0 or np.mean(scores) == 0:
                return (0, 0)  # No variability or insufficient data
            mu = np.mean(scores)
            absolute_differences = np.sum(np.abs(scores[:, None] - scores))
            gini = absolute_differences / (2 * n**2 * mu)
            caa = mu * (1 - gini)
            return (gini, caa)

        # Sort the data by player ID and date
        data = data.sort_values(by=[date_column])

        # Group by player_id and calculate rolling features
        metrics = data.groupby('player_id').apply(
            lambda group: pd.DataFrame({
                'gini_coefficient': group[score_column]
                    .expanding()
                    .apply(lambda x: gini_and_caa(x[:-1])[0], raw=False),  # Gini coefficient
                'consistency_adjusted_average': group[score_column]
                    .expanding()
                    .apply(lambda x: gini_and_caa(x[:-1])[1], raw=False)  # CAA
            }, index=group.index)
        ).reset_index(drop=True)

        # Add the calculated metrics to the original data
        data_with_metrics = pd.concat([data.reset_index(drop=True), metrics], axis=1)

        self.mw_pw_profile = data_with_metrics


    def make_all_features(self):
        """
        Run all the feature generation methods in the class.
        """
        self.get_match_type_data()
        self.process_country_and_homeaway()
        self.calculate_fantasy_scores()
        self.player_features()
        self.avg_of_player_against_opposition()
        self.add_season()
        self.batter_bowler_classification()
        self.categorize_bowling_style()
        self.sena_sub_countries()
        self.calculate_match_level_venue_stats()
        self.calculate_matches_played_before()
        self.player_features_dot_balls()
        self.get_role_factor()
        self.calculate_rolling_gini_and_caa_with_original_data()
        self.encode_preprocess()


def generate_features():
    mw_pw_profile = pd.read_csv(file_path_mw_pw, index_col=False)
    mw_overall = pd.read_csv(file_path_mw_overall, index_col=False)
    
    feature_gen = FeatureGeneration(mw_overall, mw_pw_profile, 'Test')
    feature_gen.make_all_features()
    
    out_file_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src", "data", "processed", 'final_training_file_test.csv'))
    feature_gen.mw_pw_profile.to_csv(out_file_path, index=False)