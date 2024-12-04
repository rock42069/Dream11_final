import pandas as pd
import numpy as np
import geonamescache
import pycountry
import os
from sklearn.preprocessing import LabelEncoder
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_mw_pw = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "interim", "mw_pw_profiles.csv"))
file_path_mw_overall = os.path.abspath(os.path.join(current_dir, "..", "..","src", "data", "interim", "mw_overall.csv"))

def calculate_fantasy_scores(df):
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
        player_out = row['player_out']
        stumpings = row['stumpings_done']
        fantasy_playing=0
        fantasy_batting = 0
        if match_type in ['T20', 'IT20']:
            if(match_type == 'T20'):
                fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
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
            if(match_type == 'ODI'):
                fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
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
            if(match_type == 'Test'):
                fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
            if runs_scored >= 100:
                fantasy_batting += 8
            elif runs_scored >= 50:
                fantasy_batting += 4

            if runs_scored == 0 and player_out:
                fantasy_batting -= 4

        elif match_type == 'T10':
            fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
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

        elif match_type == '6ixty':
            fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
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
        elif match_type == '6ixty':
            fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
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
        elif match_type == 'The100':
            fantasy_playing=4
            fantasy_batting = 1*runs_scored + 1*(fours_scored) + 2*(sixes_scored) + 12*stumpings
            if runs_scored >= 100:
                fantasy_batting += 20
            elif runs_scored >= 50:
                fantasy_batting += 10
            elif runs_scored >= 30:
                fantasy_batting += 5

            if runs_scored == 0 and player_out:
                fantasy_batting -= 2

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

        if match_type in ['T20', 'IT20']:
          fantasy_bowling = 25*(wickets_taken) + 8*(lbw_done+bowled_done) + 12*maidens + 8*catches_taken
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
          fantasy_bowling += run_out_throw*6 + run_out_direct*12

        elif match_type in ['ODI', 'ODM']:
            fantasy_bowling = 25*(wickets_taken) + 8*(lbw_done+bowled_done) + 4*maidens + 8*catches_taken
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
            fantasy_bowling += run_out_throw*6 + run_out_direct*12

        elif match_type in ['Test', 'MDM']:
            fantasy_bowling = 16*(wickets_taken) + 8*(lbw_done+bowled_done) + 8*catches_taken
            if wickets_taken >= 5:
                fantasy_bowling += 8
            elif wickets_taken >= 4:
                fantasy_bowling += 4
            fantasy_bowling += run_out_throw*6 + run_out_direct*12

        elif match_type == 'T10':
            fantasy_bowling = 25*(wickets_taken) + 8*(lbw_done+bowled_done) + 16*maidens + 8*catches_taken
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
            fantasy_bowling += run_out_throw*6 + run_out_direct*12

            if catches_taken >= 3:
                fantasy_bowling += 4
        elif match_type == '6ixty':
            fantasy_bowling = 25*(wickets_taken) + 8*(lbw_done+bowled_done) + 16*maidens + 8*catches_taken
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
            fantasy_bowling += run_out_throw*6 + run_out_direct*12

            if catches_taken >= 3:
                fantasy_bowling += 4

        elif match_type == 'The100':
            fantasy_bowling = 25*(wickets_taken) + 8*(lbw_done+bowled_done) + 8*catches_taken
            if wickets_taken >= 5:
                fantasy_bowling += 20
            elif wickets_taken >= 4:
                fantasy_bowling += 10
            elif wickets_taken >= 3:
                fantasy_bowling += 5
            elif wickets_taken >= 2:
                fantasy_bowling += 3
            fantasy_bowling += run_out_throw*6 + run_out_direct*12

            if catches_taken >= 3:
                fantasy_bowling += 4

        df.at[index, 'fantasy_score_bowling'] = fantasy_bowling
        df.at[index, 'fantasy_score_total'] = fantasy_playing
    df['fantasy_score_total'] = df['fantasy_score_batting'] + df['fantasy_score_bowling']+df['fantasy_score_total']
    return df

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
    '''
    Adds a hemisphere column to the dataframe based on the country.
    '''
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
        Handles T10, 6ixty, The100, T20, IT20, ODI, and ODM match types.
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

    def binaryclassification(self):
        final_df = self.mw_pw_profile
        match_id_counts = final_df['match_id'].value_counts()

        valid_match_ids = match_id_counts[match_id_counts == 22].index

        # Filter the original DataFrame
        filtered_final_df = final_df[final_df['match_id'].isin(valid_match_ids)]

        # Initialize the 'selected' column with 0
        filtered_final_df['selected'] = 0

        # For each 'match_id', sort by 'fantasy_score_total' and mark the top 11 rows
        for match_id in valid_match_ids:
            match_rows = filtered_final_df[filtered_final_df['match_id'] == match_id]
            top_11_indices = match_rows.sort_values(by='fantasy_score_total', ascending=False).head(11).index
            filtered_final_df.loc[top_11_indices, 'selected'] = 1
        
        # Reset index if needed
        filtered_final_df = filtered_final_df.reset_index(drop=True)
        self.mw_pw_profile = filtered_final_df
    
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
        self.binaryclassification()
        self.encode_preprocess()

def main_test():
    mw_pw_profile = pd.read_csv(file_path_mw_pw, index_col=False)
    mw_overall = pd.read_csv(file_path_mw_overall, index_col=False)
    
    feature_gen = FeatureGeneration(mw_overall, mw_pw_profile, 'Test')
    feature_gen.make_all_features()
    
    out_file_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src", "data", "processed", 'final_training_file_test.csv'))
    feature_gen.mw_pw_profile.to_csv(out_file_path, index=False)

############################################################################################################################################################################
############################################################################################################################################################################
#############################################       T20  FEATURES       ##################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################


class FeatureEngineering_t20:

    def preprocessdf(self, df):
        df['start_date'] = pd.to_datetime(df['start_date'])
        df = df.sort_values(by='start_date').reset_index(drop=True)
        return df

    def encode_with_occurrences(self, df, column_name):
        value_counts = df[column_name].value_counts().to_dict()
        df[column_name] = df[column_name].map(value_counts)
        return df

    def encode_with_occurrencesnew(self, df, column_name, newcolumnname):
        value_counts = df[column_name].value_counts().to_dict()
        df[newcolumnname] = df[column_name].map(value_counts)
        return df

    def split(self, df, date):
        train = df[df['start_date'] <= pd.to_datetime(date)]
        test = df[df['start_date'] > pd.to_datetime(date)]
        return train, test

    def one_hot_encode(self, X, column_name):
        unique_values = np.unique(X[column_name])
        one_hot_dict = {f"{column_name}_{unique_value}": (X[column_name] == unique_value).astype(int) for unique_value in unique_values}
        X = X.drop(columns=[column_name])
        for col_name, col_data in one_hot_dict.items():
            X[col_name] = col_data
        return X

    def Labelencode(self, X, col):
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        return X

    def get_top_min_scores(self, data, top_n=10):
        scores = [(key, regressor, score) for key, regressors in data.items() for regressor, score in regressors.items()]
        scores_sorted = sorted(scores, key=lambda x: x[2])
        return scores_sorted[:top_n]

    def calculate_centuries(self, runs_scored):
        return (runs_scored >= 100).sum()

    def calculate_half_centuries(self, runs_scored):
        return ((runs_scored < 100) & (runs_scored >= 50)).sum()

    def prepare_data(self, data, cutoff_date):
        data['start_date'] = pd.to_datetime(data['start_date'])
        data = data.sort_values(['player_id', 'match_type', 'start_date'])
        return data[data['start_date'] < cutoff_date]

    def calculate_rolling_batting_stats_with_context(self, group, n1, n2, n3, min_balls=20):
        group = group.sort_values('start_date')
        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()
        for n in [n1, n2, n3]:
            runs = group['runs_scored'].shift().rolling(n, min_periods=1).sum()
            balls = group['balls_faced'].shift().rolling(n, min_periods=1).sum()
            player_out = group['player_out'].shift().rolling(n, min_periods=1).sum()
            boundary_runs = (group['fours_scored'].shift().rolling(n, min_periods=1).sum() * 4 +
                             group['sixes_scored'].shift().rolling(n, min_periods=1).sum() * 6)
            singles = runs - boundary_runs
            if n == n1:
                group['batting_average_n1'] = runs / player_out.replace(0, np.nan)
            elif n == n2:
                group['batting_average_n2'] = runs / player_out.replace(0, np.nan)
            elif n == n3:
                group['batting_average_n3'] = runs / player_out.replace(0, np.nan)
            if n == n1:
                group['strike_rate_n1'] = np.where(balls >= min_balls, (runs / balls) * 100, np.nan)
            elif n == n2:
                group['strike_rate_n2'] = np.where(balls >= min_balls, (runs / balls) * 100, np.nan)
            elif n == n3:
                group['strike_rate_n3'] = np.where(balls >= min_balls, (runs / balls) * 100, np.nan)
            if n == n1:
                group['boundary_percentage_n1'] = np.where(runs > 0, (boundary_runs / runs) * 100, np.nan)
                group['strike_rotation_percentage_n1'] = np.where(balls > 0, (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, np.nan)
            elif n == n2:
                group['boundary_percentage_n2'] = np.where(runs > 0, (boundary_runs / runs) * 100, np.nan)
                group['strike_rotation_percentage_n2'] = np.where(balls > 0, (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, np.nan)
            elif n == n3:
                group['boundary_percentage_n3'] = np.where(runs > 0, (boundary_runs / runs) * 100, np.nan)
                group['strike_rotation_percentage_n3'] = np.where(balls > 0, (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, np.nan)
            if n == n1:
                group['avg_batting_average_n1'] = ema(group['batting_average_n1'], span=n)
                group['avg_strike_rate_n1'] = ema(group['strike_rate_n1'], span=n)
            elif n == n2:
                group['avg_batting_average_n2'] = ema(group['batting_average_n2'], span=n)
                group['avg_strike_rate_n2'] = ema(group['strike_rate_n2'], span=n)
            elif n == n3:
                group['avg_batting_average_n3'] = ema(group['batting_average_n3'], span=n)
                group['avg_strike_rate_n3'] = ema(group['strike_rate_n3'], span=n)
        return group

    def calculate_ema(self, series, window):
        alpha = 2 / (window + 1)
        ema = series.ewm(span=window, adjust=False).mean()
        return ema

    def calculate_rolling_bowling_stats_with_ema(self, group, n1, n2, n3):
        group = group.sort_values('start_date')
        group['runs_n1'] = self.calculate_ema(group['runs_conceded'], n1)
        group['wickets_n1'] = self.calculate_ema(group['wickets_taken'], n1)
        group['balls_n1'] = self.calculate_ema(group['balls_bowled'], n1)
        group['bowling_average_n1'] = group['runs_n1'] / group['wickets_n1'].replace(0, np.nan)
        group['economy_rate_n1'] = group['runs_n1'] / (group['balls_n1'] / 6)
        group['bowling_strike_rate_n1'] = group['balls_n1'] / group['wickets_n1'].replace(0, np.nan)
        group['runs_n2'] = self.calculate_ema(group['runs_conceded'], n2)
        group['wickets_n2'] = self.calculate_ema(group['wickets_taken'], n2)
        group['balls_n2'] = self.calculate_ema(group['balls_bowled'], n2)
        group['bowling_average_n2'] = group['runs_n2'] / group['wickets_n2'].replace(0, np.nan)
        group['economy_rate_n2'] = group['runs_n2'] / (group['balls_n2'] / 6)
        group['bowling_strike_rate_n2'] = group['balls_n2'] / group['wickets_n2'].replace(0, np.nan)
        group['runs_n3'] = self.calculate_ema(group['runs_conceded'], n3)
        group['wickets_n3'] = self.calculate_ema(group['wickets_taken'], n3)
        group['balls_n3'] = self.calculate_ema(group['balls_bowled'], n3)
        group['bowling_average_n3'] = group['runs_n3'] / group['wickets_n3'].replace(0, np.nan)
        group['economy_rate_n3'] = group['runs_n3'] / (group['balls_n3'] / 6)
        group['bowling_strike_rate_n3'] = group['balls_n3'] / group['wickets_n3'].replace(0, np.nan)
        group['CBR'] = self.calculate_cbr(group['bowling_average_n2'], group['economy_rate_n2'], group['bowling_strike_rate_n2'])
        group['CBR2'] = self.calculate_cbr(group['bowling_average_n3'], group['economy_rate_n3'], group['bowling_strike_rate_n3'])
        group['fielding_points'] = (group['catches_taken'].shift().rolling(n3, min_periods=1).sum() * 8 +
                                    group['stumpings_done'].shift().rolling(n3, min_periods=1).sum() * 12 +
                                    group['run_out_direct'].shift().rolling(n3, min_periods=1).sum() * 12 +
                                    group['run_out_throw'].shift().rolling(n3, min_periods=1).sum() * 6)
        return group

    def calculate_cbr(self, avg, econ, sr):
        avg = np.where(avg > 0, np.log1p(avg), np.inf)
        econ = np.where(econ > 0, np.log1p(econ), np.inf)
        sr = np.where(sr > 0, np.log1p(sr), np.inf)
        return (avg * econ * sr) / (avg + econ + sr)

    def calculate_centuries_and_half_centuries(self, group):
        group = group.sort_values('start_date')
        group['centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_centuries)
        group['half_centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_half_centuries)
        return group

    def calculate_additional_stats(self, group, n):
        group = group.sort_values('start_date')
        group['wickets_in_n_matches'] = group['wickets_taken'].shift().rolling(n, min_periods=1).sum()
        group['wickets_in_n_matches'] = group['wickets_in_n_matches'] / n
        group['total_overs_throwed'] = (group['balls_bowled'].shift() / group['balls_per_over']).rolling(n, min_periods=1).sum()
        group['highest_runs'] = group['runs_scored'].shift().expanding(min_periods=1).max()
        group['highest_wickets'] = group['wickets_taken'].shift().expanding(min_periods=1).max()
        group['four_wicket_hauls_n'] = (group['wickets_taken'] >= 4).shift().rolling(n, min_periods=1).sum()
        return group

    def calculate_rolling_fantasy_score_features(self, group):
        group['avg_fantasy_score_1'] = group['fantasy_score_total'].shift().rolling(1, min_periods=1).mean()
        group['avg_fantasy_score_5'] = group['fantasy_score_total'].shift().rolling(5, min_periods=1).mean()
        group['avg_fantasy_score_10'] = group['fantasy_score_total'].shift().rolling(10, min_periods=1).mean()
        group['avg_fantasy_score_15'] = group['fantasy_score_total'].shift().rolling(15, min_periods=1).mean()
        group['avg_fantasy_score_20'] = group['fantasy_score_total'].shift().rolling(20, min_periods=1).mean()
        return group

    def calculate_rolling_ducks(self, group, n):
        group['ducks'] = ((group['runs_scored'] == 0) & (group['player_out'] == 1)).astype(int)
        group['rolling_ducks'] = group['ducks'].shift().rolling(n, min_periods=1).sum()
        group['rolling_ducks'] = group['rolling_ducks'] ** 2
        return group

    def calculate_rolling_maidens(self, group, n):
        group['rolling_maidens'] = group['maidens'].shift().rolling(n, min_periods=1).sum()
        return group

    def calculate_alpha_batsmen_score_for_odis(self, group, n):
        group = group.sort_values('start_date')
        group['avg_runs_scored'] = group['runs_scored'].shift().rolling(n, min_periods=1).mean()
        group['avg_strike_rate'] = group['strike_rate_n1'].shift().rolling(n, min_periods=1).mean()
        group['avg_half_centuries'] = group['half_centuries_cumsum'].shift().rolling(n, min_periods=1).sum()
        group['avg_centuries'] = group['centuries_cumsum'].shift().rolling(n, min_periods=1).sum()
        group['avg_rolling_ducks'] = group['rolling_ducks'].shift().rolling(n, min_periods=1).sum()
        group['boundary_runs'] = (group['fours_scored'].shift() * 4) + (group['sixes_scored'].shift() * 6)
        group['singles_scored'] = group['runs_scored'].shift() - group['boundary_runs']
        group['strike_rotation_percentage'] = (group['singles_scored'] / group['balls_faced'].shift() - group['fours_scored'].shift() - group['sixes_scored'].shift()) * 100
        group['avg_strike_rotation_percentage'] = group['strike_rotation_percentage'].shift().rolling(n, min_periods=1).mean()
        group.fillna(0, inplace=True)
        group['α_batsmen_score'] = (
            0.25 * group['avg_runs_scored'] +
            0.10 * group['avg_strike_rate'] +
            0.20 * group['avg_centuries'] +
            0.15 * group['avg_half_centuries'] +
            1.5 * group['avg_rolling_ducks'] +
            0.15 * group['avg_strike_rotation_percentage']
        )
        return group

    def calculate_alpha_bowler_score(self, group, n):
        group = group.sort_values('start_date')
        group['avg_wickets_taken'] = group['wickets_taken'].shift().rolling(n, min_periods=1).mean()
        group['avg_bowling_average'] = group['bowling_average_n1'].shift().rolling(n, min_periods=1).mean()
        group['avg_bowling_strike_rate'] = group['bowling_strike_rate_n1'].shift().rolling(n, min_periods=1).mean()
        group['avg_economy_rate'] = group['economy_rate_n1'].shift().rolling(n, min_periods=1).mean()
        group['avg_maidens'] = group['rolling_maidens'].shift().rolling(n, min_periods=1).sum()
        group.fillna(0, inplace=True)
        group['α_bowler_score'] = (
            0.35 * group['avg_wickets_taken'] +
            0.25 * group['avg_bowling_strike_rate'] +
            0.20 * group['avg_economy_rate'] +
            0.10 * group['avg_maidens'] -
            0.10 * group['avg_bowling_average']
        )
        return group

    def assign_rating_score(self,group):
        """
        Assign batsman and bowler ratings based on predefined ranges.
        Parameters:
        - group: DataFrame containing player performance data with 'α_batsmen_score' and 'α_bowler_score'.
        Returns:
        - group: DataFrame with 'batsman_rating' and 'bowler_rating' added.
        """
        # Define batsman rating ranges
        batsman_ranges = {
            (0, 5): 0,
            (5, 15): 4,
            (15, 25): 9,
            (25, 35): 16,
            (35, 45): 25,
            (45, 55): 49,
            (55, float('inf')): 81
        }

        # Define bowler rating ranges
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

        # Helper function to assign rating based on ranges
        def get_rating(score, ranges):
            for (lower, upper), rating in ranges.items():
                if lower <= score < upper:
                    return rating
            return 0  # Default in case no range matches

        # Apply batsman and bowler ratings
        group['batsman_rating'] = group['α_batsmen_score'].apply(lambda x: get_rating(x, batsman_ranges))
        group['bowler_rating'] = group['α_bowler_score'].apply(lambda x: get_rating(x, bowler_ranges))

        return group
    def run_30_to_50(self, group):
        group = group.sort_values('start_date')
        group['cumulative_30s'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_30s)
        group['conversion_30_to_50'] = group.apply(lambda x: (x['half_centuries_cumsum'] / x['cumulative_30s']) if x['cumulative_30s'] != 0 else 0, axis=1)
        return group

    def calculate_30s(self, runs_scored):
        """Calculate the total number of 30s."""
        return ((runs_scored >= 30) & (runs_scored < 50)).sum()

    def longtermfeatures(self, group):
        """Calculate long-term career features for batting and bowling."""
        group = group.sort_values('start_date')
        
        # Batting features
        group['longterm_avg_runs'] = group['runs_scored'].shift().expanding().mean()
        group['longterm_var_runs'] = np.sqrt(group['runs_scored'].shift().expanding().var())
        group['longterm_avg_strike_rate'] = (
            (group['runs_scored'].shift().expanding().sum()) /
            (group['balls_faced'].shift().expanding().sum()) * 100
        )

        # Bowling features
        group['longterm_avg_wickets_per_match'] = group['wickets_taken'].shift().expanding().mean()
        group['longterm_var_wickets_per_match'] = np.sqrt(group['wickets_taken'].shift().expanding().var())
        group['longterm_avg_economy_rate'] = (
            (group['runs_conceded'].shift().expanding().sum()) /
            ((group['balls_bowled'].shift().expanding().sum()) / group['balls_per_over'].iloc[0])
        )
        group['longterm_total_matches_of_type'] = group.groupby('match_type').cumcount()
        return group

    def order_seen(self, group):
        group['order_seen_mode'] = group['order_seen'].shift().expanding().apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
        return group

    def player_features(self, data, n1, n2, n3):
        """Main function to calculate player features using helper functions."""
        cutoff_date = pd.to_datetime('2024-12-06')
        data = self.prepare_data(data, cutoff_date)

        feature_data = []

        for (name, match_type), group in data.groupby(['player_id', 'match_type']):
            group = group.sort_values('start_date')
            group = self.longtermfeatures(group)
            group = self.calculate_rolling_batting_stats_with_context(group, n1, n2, n3)
            group = self.calculate_rolling_bowling_stats_with_ema(group, n1, n2, n3)
            group = self.calculate_centuries_and_half_centuries(group)
            group = self.calculate_additional_stats(group, n3)
            group = self.calculate_rolling_fantasy_score_features(group)
            group = self.calculate_rolling_ducks(group, n3)
            group = self.calculate_rolling_maidens(group, n3)
            group = self.calculate_alpha_batsmen_score_for_odis(group, n3)
            group = self.calculate_alpha_bowler_score(group, n3)
            group = self.assign_rating_score(group)
            group = self.order_seen(group)
            group = self.run_30_to_50(group)
            feature_data.append(group[['player_id', 'match_id', 'match_type', 'start_date',
                                    'batting_average_n1', 'strike_rate_n1', 'boundary_percentage_n1',
                                    'batting_average_n2', 'strike_rate_n2', 'boundary_percentage_n2',
                                    'batting_average_n3', 'strike_rate_n3', 'boundary_percentage_n3',
                                    'centuries_cumsum', 'half_centuries_cumsum',
                                    'avg_runs_scored', 'avg_strike_rate', 'avg_half_centuries', 'avg_centuries',
                                    'avg_rolling_ducks', 'strike_rotation_percentage', 'avg_strike_rotation_percentage', 'conversion_30_to_50',
                                    'economy_rate_n1', 'economy_rate_n2', 'economy_rate_n3',
                                    'wickets_in_n_matches', 'total_overs_throwed',
                                    'bowling_average_n1', 'bowling_strike_rate_n1',
                                    'bowling_average_n2', 'bowling_strike_rate_n2',
                                    'bowling_average_n3', 'bowling_strike_rate_n3',
                                    'CBR', 'CBR2',
                                    'fielding_points', 'four_wicket_hauls_n', 'highest_runs', 'highest_wickets',
                                    'order_seen_mode',
                                    'longterm_avg_runs', 'longterm_var_runs', 'longterm_avg_strike_rate',
                                    'longterm_avg_wickets_per_match', 'longterm_var_wickets_per_match',
                                    'longterm_avg_economy_rate', 'longterm_total_matches_of_type',
                                    'avg_fantasy_score_1', 'avg_fantasy_score_5', 'avg_fantasy_score_10',
                                    'avg_fantasy_score_15', 'avg_fantasy_score_20',
                                    'rolling_ducks', 'rolling_maidens',
                                    'α_batsmen_score', 'α_bowler_score',
                                    'batsman_rating', 'bowler_rating']])

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)  # Reset index and drop old index
        data = data.merge(result_df, on=['player_id', 'match_type', 'match_id', 'start_date'], how='left')
        data = data.sort_values(by='start_date').reset_index(drop=True)
        return data

    def calculate_matches_played_before(self, df):
        df = df.sort_values(by=['player_id', 'match_type', 'start_date'])
        df['longterm_total_matches_of_type'] = df.groupby(['player_id', 'match_type']).cumcount()
        return df

    def categorize_bowling_style(self, style):
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

    def calculate_rolling_fantasy_score(self, group):
        group['avg_fantasy_score_5_bowler'] = group['fantasy_score_bowling'].shift().rolling(5, min_periods=1).mean()
        group['avg_fantasy_score_5_batting'] = group['fantasy_score_batting'].shift().rolling(5, min_periods=1).mean()
        return group

    def add_matchup_features_with_bowler_bonus(self, group):
        group['batter_penalty'] = 0.0
        group['bowler_bonus'] = 0.0

        for idx, row in group.iterrows():
            batter_penalty = 0.0
            bowler_bonus = 0.0

            same_match_players = group[group['match_id'] == row['match_id']]
            opposing_players = same_match_players[same_match_players['player_team'] == row['opposition_team']]

            if 'Bowler' not in row['playing_role'] if isinstance(row['playing_role'], str) else True:
                if row['batting_style'] == 'Right hand Bat' and pd.notna(row['playing_role']) and isinstance(row['playing_role'], str) and 'Middle order Batter' in row['playing_role']:
                    n = opposing_players['bowling_style'].str.contains('Wrist spin', case=False, na=False).sum()
                    avg_fantasy_score = opposing_players.loc[
                        opposing_players['bowling_style'].str.contains('Wrist spin', case=False, na=False),
                        'avg_fantasy_score_5_bowler'
                    ].mean()
                    batter_penalty -= n * avg_fantasy_score if pd.notna(avg_fantasy_score) else 0

                if any(role in row['playing_role'] for role in ['Opening Batter', 'Top Order Batter']) if isinstance(row['playing_role'], str) else False:
                    n = opposing_players['bowling_style'].str.contains(
                        'Left arm fast|Left arm Medium fast|Left arm Fast medium', case=False, na=False
                    ).sum()
                    avg_fantasy_score = opposing_players.loc[
                        opposing_players['bowling_style'].str.contains(
                            'Left arm fast|Left arm Medium fast|Left arm Fast medium', case=False, na=False
                        ),
                        'avg_fantasy_score_5_bowler'
                    ].mean()
                    batter_penalty -= n * avg_fantasy_score if pd.notna(avg_fantasy_score) else 0

                if row['batting_style'] == 'Left hand Bat' and any(role in row['playing_role'] for role in ['Top order Batter', 'Batter', 'Batting Allrounder']) if isinstance(row['playing_role'], str) else False:
                    n = opposing_players['bowling_style'].str.contains('Orthodox', case=False, na=False).sum()
                    avg_fantasy_score = opposing_players.loc[
                        opposing_players['bowling_style'].str.contains('Orthodox', case=False, na=False),
                        'avg_fantasy_score_5_bowler'
                    ].mean()
                    batter_penalty -= n * avg_fantasy_score if pd.notna(avg_fantasy_score) else 0

            if 'Bowler' in row['playing_role'] if isinstance(row['playing_role'], str) else False:
                if isinstance(row['bowling_style'], str) and 'Left arm Orthodox' in row['bowling_style']:
                    n = opposing_players['batting_style'].str.contains('Left hand Bat', case=False, na=False).sum()
                    avg_fantasy_score = opposing_players.loc[
                        opposing_players['batting_style'].str.contains('Left hand Bat', case=False, na=False),
                        'avg_fantasy_score_5_batting'
                    ].mean()
                    bowler_bonus += n * avg_fantasy_score

                if row['bowling_style'] == 'Wrist spin':
                    n = opposing_players[
                        (opposing_players['batting_style'] == 'Right hand Bat') &
                        (opposing_players['playing_role'].str.contains('Middle order Batter', case=False, na=False))
                    ].shape[0]
                    avg_fantasy_score = opposing_players.loc[
                        (opposing_players['batting_style'] == 'Right hand Bat') &
                        (opposing_players['playing_role'].str.contains('Middle order Batter', case=False, na=False)),
                        'avg_fantasy_score_5_batting'
                    ].mean()
                    bowler_bonus += n * avg_fantasy_score

                if row['bowling_style'] in ['Left arm fast', 'Left arm Medium fast', 'Left arm Fast medium']:
                    n = opposing_players[
                        opposing_players['playing_role'].apply(lambda x: isinstance(x, str) and ('Opening Batter' in x or 'Batter' in x))
                    ].shape[0]
                    avg_fantasy_score = opposing_players.loc[
                        opposing_players['playing_role'].apply(lambda x: isinstance(x, str) and ('Opening Batter' in x or 'Batter' in x)),
                        'avg_fantasy_score_5_batting'
                    ].mean()
                    bowler_bonus += n * avg_fantasy_score

                if isinstance(row['bowling_style'], str) and 'Orthodox' in row['bowling_style']:
                    n = opposing_players[
                        (opposing_players['batting_style'] == 'Left hand Bat') &
                        (opposing_players['playing_role'].apply(lambda x: isinstance(x, str) and 'Top order Batter' in x))
                    ].shape[0]
                    avg_fantasy_score = opposing_players.loc[
                        (opposing_players['batting_style'] == 'Left hand Bat') &
                        (opposing_players['playing_role'].apply(lambda x: isinstance(x, str) and 'Top order Batter' in x)),
                        'avg_fantasy_score_5_batting'
                    ].mean()
                    bowler_bonus += n * avg_fantasy_score

            group.at[idx, 'batter_penalty'] = batter_penalty
            group.at[idx, 'bowler_bonus'] = bowler_bonus

        return group

    def matchup_features_with_bonus(self, data):
        feature_data = []

        for match_id, group in data.groupby('match_id'):
            group = self.calculate_rolling_fantasy_score(group)
            group = self.add_matchup_features_with_bowler_bonus(group)
            feature_data.append(group[['player_id', 'batter_penalty', 'bowler_bonus']])

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)
        return result_df

    def target_encode(self, df, column, target, smoothing=1):
        global_mean = df[target].mean()
        agg = df.groupby(column)[target].agg(['mean', 'count'])
        smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - 1) / smoothing))
        agg['smoothed_mean'] = global_mean * (1 - smoothing_factor) + agg['mean'] * smoothing_factor
        encoded_series = df[column].map(agg['smoothed_mean'])
        return encoded_series

    def select_top_players_by_match(self, df):
        match_id_counts = df['match_id'].value_counts()
        valid_match_ids = match_id_counts[match_id_counts == 22].index
        filtered_df = df[df['match_id'].isin(valid_match_ids)].copy()
        filtered_df['selected'] = 0

        for match_id in valid_match_ids:
            match_rows = filtered_df[filtered_df['match_id'] == match_id]
            top_11_indices = match_rows.sort_values(by='fantasy_score_total', ascending=False).head(11).index
            filtered_df.loc[top_11_indices, 'selected'] = 1

        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df
    def binaryclassification(self,final_df):
        match_id_counts = final_df['match_id'].value_counts()

        valid_match_ids = match_id_counts[match_id_counts == 22].index

        # Filter the original DataFrame
        filtered_final_df = final_df[final_df['match_id'].isin(valid_match_ids)]

        # Initialize the 'selected' column with 0
        filtered_final_df['selected'] = 0

        # For each 'match_id', sort by 'fantasy_score_total' and mark the top 11 rows
        for match_id in valid_match_ids:
            match_rows = filtered_final_df[filtered_final_df['match_id'] == match_id]
            top_11_indices = match_rows.sort_values(by='fantasy_score_total', ascending=False).head(11).index
            filtered_final_df.loc[top_11_indices, 'selected'] = 1
        
        # Reset index if needed
        filtered_final_df = filtered_final_df.reset_index(drop=True)
        return filtered_final_df
    
    def generate_features_t20(self):
        
        par_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src", "data", "interim", "mw_pw_profiles.csv"))

        df = pd.read_csv(par_path, index_col=False)
        df = df[(df['match_type'] == 'T20') | (df['match_type'] == 'IT20')]

        df = self.preprocessdf(df)
        df = calculate_fantasy_scores(df)
        df = self.player_features(df, 3, 7, 12)

        df['avg_against_opposition'] = (
            df.groupby(['player_id', 'opposition_team'])['fantasy_score_total']
            .apply(lambda x: x.shift().expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        df = self.binaryclassification(df)

        df = self.preprocessdf(df)
        df = self.calculate_matches_played_before(df)
        df = df.sort_values(by=['start_date']).reset_index(drop=True)

        df['batter'] = np.where(
            pd.isna(df['playing_role']),
            0,
            np.where(
                df['playing_role'].str.contains(r'Batter|Wicketkeeper|Batting Allrounder', case=False, na=False),
                1,
                0
            )
        )

        df['bowler'] = np.where(
            pd.isna(df['playing_role']),
            0,
            np.where(
                df['playing_role'].str.contains(r'Bowler|Bowling Allrounder', case=False, na=False),
                1,
                0
            )
        )

        df['bowling_style'] = df['bowling_style'].apply(self.categorize_bowling_style)
        df['bowling_style'] = self.target_encode(df, 'bowling_style', 'fantasy_score_total')
        # df = self.select_top_players_by_match(df)
        df = self.preprocessdf(df)

        out_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src", "data", "processed", "final_training_file_t20.csv"))
        df.to_csv(out_path, index=False)


def main_t20():
    feature_generator = FeatureEngineering_t20()
    feature_generator.generate_features_t20()

############################################################################################################################################################################
############################################################################################################################################################################
#############################################       ODI  FEATURES       ##################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

# import time
import pandas as pd
import numpy as np
import geonamescache
import pycountry

class FeatureGeneratorODI:

    def calculate_centuries(self, runs_scored):
        """Calculate the total number of centuries."""
        return (runs_scored >= 100).sum()

    def calculate_half_centuries(self, runs_scored):
        """Calculate the total number of half-centuries (50 <= runs < 100)."""
        return ((runs_scored < 100) & (runs_scored >= 50)).sum()

    def prepare_data(self, data,cutoff_date):
        """Prepare and sort data, set date format, and apply cutoff date."""
        data['start_date'] = pd.to_datetime(data['start_date'])  # Ensure 'start_date' is in datetime format
        data = data.sort_values(['player_id', 'match_type', 'start_date'])
        return data[data['start_date'] < cutoff_date]

    def calculate_rolling_batting_stats_with_context(self, group, n1, n2, n3):
        """Calculate enhanced batting stats for ODIs with contextual adjustments."""
        group = group.sort_values('start_date')

        # Define helper function for exponential moving average
        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        # Rolling aggregates for n1, n2, n3
        for n in [n1, n2, n3]:
            # Basic aggregates
            runs = group['runs_scored'].shift().rolling(n, min_periods=1).sum()
            balls = group['balls_faced'].shift().rolling(n, min_periods=1).sum()
            player_out = group['player_out'].shift().rolling(n, min_periods=1).sum()
            boundary_runs = (group['fours_scored'].shift().rolling(n, min_periods=1).sum() * 4 +
                             group['sixes_scored'].shift().rolling(n, min_periods=1).sum() * 6)
            singles = runs - boundary_runs

            # Contextual Batting Average
            if n == n1:
                group['batting_average_n1'] = runs / player_out.replace(0, np.nan)
            elif n == n2:
                group['batting_average_n2'] = runs / player_out.replace(0, np.nan)
            elif n == n3:
                group['batting_average_n3'] = runs / player_out.replace(0, np.nan)

            # Weighted Strike Rate
            if n == n1:
                group['strike_rate_n1'] = np.where(
                    balls >= 20, 
                    (runs / balls) * 100, 
                    np.nan
                )
            elif n == n2:
                group['strike_rate_n2'] = np.where(
                    balls >= 20, 
                    (runs / balls) * 100, 
                    np.nan
                )
            elif n == n3:
                group['strike_rate_n3'] = np.where(
                    balls >= 20, 
                    (runs / balls) * 100, 
                    np.nan
                )

            # Boundary Percentage + Strike Rotation
            if n == n1:
                group['boundary_percentage_n1'] = np.where(
                    runs > 0, 
                    (boundary_runs / runs) * 100, 
                    np.nan
                )
                group['strike_rotation_percentage_n1'] = np.where(
                    balls > 0, 
                    (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, 
                    np.nan
                )
            elif n == n2:
                group['boundary_percentage_n2'] = np.where(
                    runs > 0, 
                    (boundary_runs / runs) * 100, 
                    np.nan
                )
                group['strike_rotation_percentage_n2'] = np.where(
                    balls > 0, 
                    (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, 
                    np.nan
                )
            elif n == n3:
                group['boundary_percentage_n3'] = np.where(
                    runs > 0, 
                    (boundary_runs / runs) * 100, 
                    np.nan
                )
                group['strike_rotation_percentage_n3'] = np.where(
                    balls > 0, 
                    (singles / (balls - group['fours_scored'] - group['sixes_scored'])) * 100, 
                    np.nan
                )

            # Exponential Moving Averages for Recent Form
            if n == n1:
                group['avg_batting_average_n1'] = ema(group['batting_average_n1'], span=n)
                group['avg_strike_rate_n1'] = ema(group['strike_rate_n1'], span=n)
            elif n == n2:
                group['avg_batting_average_n2'] = ema(group['batting_average_n2'], span=n)
                group['avg_strike_rate_n2'] = ema(group['strike_rate_n2'], span=n)
            elif n == n3:
                group['avg_batting_average_n3'] = ema(group['batting_average_n3'], span=n)
                group['avg_strike_rate_n3'] = ema(group['strike_rate_n3'], span=n)

        return group

    def calculate_ema(self, series, window):
        """Compute Exponential Moving Average (EMA) for a given series."""
        alpha = 2 / (window + 1)
        ema = series.ewm(span=window, adjust=False).mean()
        return ema

    def calculate_rolling_bowling_stats_with_ema(self, group, n1, n2, n3):
        """
        Calculate bowling averages, economy rates, strike rates, and an updated CBR using EMA (Exponential Moving Average).
        """
        group = group.sort_values('start_date')

        # EMA for n1
        group['runs_n1'] = self.calculate_ema(group['runs_conceded'].shift(), n1)
        group['wickets_n1'] = self.calculate_ema(group['wickets_taken'].shift(), n1)
        group['balls_n1'] = self.calculate_ema(group['balls_bowled'].shift(), n1)

        group['bowling_average_n1'] = group['runs_n1'] / group['wickets_n1'].replace(0, np.nan)
        group['economy_rate_n1'] = group['runs_n1'] / (group['balls_n1'] / 6)  # Economy rate = runs / overs
        group['bowling_strike_rate_n1'] = group['balls_n1'] / group['wickets_n1'].replace(0, np.nan)

        # EMA for n2
        group['runs_n2'] = self.calculate_ema(group['runs_conceded'].shift(), n2)
        group['wickets_n2'] = self.calculate_ema(group['wickets_taken'].shift(), n2)
        group['balls_n2'] = self.calculate_ema(group['balls_bowled'].shift(), n2)

        group['bowling_average_n2'] = group['runs_n2'] / group['wickets_n2'].replace(0, np.nan)
        group['economy_rate_n2'] = group['runs_n2'] / (group['balls_n2'] / 6)  # Economy rate = runs / overs
        group['bowling_strike_rate_n2'] = group['balls_n2'] / group['wickets_n2'].replace(0, np.nan)

        # EMA for n3
        group['runs_n3'] = self.calculate_ema(group['runs_conceded'].shift(), n3)
        group['wickets_n3'] = self.calculate_ema(group['wickets_taken'].shift(),n3)
        group['balls_n3'] = self.calculate_ema(group['balls_bowled'].shift(),n3)

        group['bowling_average_n3'] = group['runs_n3'] / group['wickets_n3'].replace(0, np.nan)
        group['economy_rate_n3'] = group['runs_n3'] / (group['balls_n3'] / 6)  # Economy rate = runs / overs
        group['bowling_strike_rate_n3'] = group['balls_n3'] / group['wickets_n3'].replace(0, np.nan)

        # Updated CBR formula with EMA
        def calculate_cbr(avg, econ, sr):
            # Normalize and handle extreme values
            avg = np.where(avg > 0, np.log1p(avg), np.inf)
            econ = np.where(econ > 0, np.log1p(econ), np.inf)
            sr = np.where(sr > 0, np.log1p(sr), np.inf)

            # Compute CBR: Simplified and robust
            return (avg * econ * sr) / (avg + econ + sr)

        group['CBR'] = calculate_cbr(
            group['bowling_average_n2'], group['economy_rate_n2'], group['bowling_strike_rate_n2']
        )
        group['CBR2'] = calculate_cbr(
            group['bowling_average_n3'], group['economy_rate_n3'], group['bowling_strike_rate_n3']
        )

        group['fielding_points'] = (
            group['catches_taken'].shift().rolling(n3, min_periods=1).sum() * 8 +
            group['stumpings_done'].shift().rolling(n3, min_periods=1).sum() * 12 +
            group['run_out_direct'].shift().rolling(n3, min_periods=1).sum() * 12 +
            group['run_out_throw'].shift().rolling(n3, min_periods=1).sum() * 6
        )

        return group

    def calculate_centuries_and_half_centuries(self, group):
        """Calculate cumulative centuries and half-centuries up to each date."""
        group = group.sort_values('start_date')
        group['centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_centuries)
        group['half_centuries_cumsum'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_half_centuries)
        return group

    def calculate_additional_stats(self, group, n3):
        """Calculate additional cumulative and rolling stats for wickets and overs bowled."""
        group = group.sort_values('start_date')
        group['wickets_in_n_matches'] = group['wickets_taken'].shift().rolling(n3, min_periods=1).sum()
        group['wickets_in_n_matches'] = group['wickets_in_n_matches'] / n3
        group['total_overs_throwed'] = (group['balls_bowled'].shift() / group['balls_per_over']).rolling(n3, min_periods=1).sum()

        # Shift the values down by 1 to exclude the current match, then calculate the cumulative max
        group['highest_runs'] = group['runs_scored'].shift().expanding(min_periods=1).max()
        group['highest_wickets'] = group['wickets_taken'].shift().expanding(min_periods=1).max()

        group['four_wicket_hauls_n'] = (group['wickets_taken'] >= 4).shift().rolling(n3, min_periods=1).sum()
        return group

    def calculate_rolling_fantasy_score(self, group):
        """Calculate the rolling average of fantasy scores."""
        group['avg_fantasy_score_1'] = group['fantasy_score_total'].shift().rolling(1, min_periods=1).mean()
        group['avg_fantasy_score_5'] = group['fantasy_score_total'].shift().rolling(5, min_periods=1).mean()
        group['avg_fantasy_score_10'] = group['fantasy_score_total'].shift().rolling(10, min_periods=1).mean()
        group['avg_fantasy_score_15'] = group['fantasy_score_total'].shift().rolling(15, min_periods=1).mean()
        group['avg_fantasy_score_20'] = group['fantasy_score_total'].shift().rolling(20, min_periods=1).mean()

        return group

    def calculate_rolling_ducks(self, group, n3):
        """Calculate the rolling sum of ducks (runs_scored == 0 and player_out == 1) over the last n matches."""
        group['ducks'] = ((group['runs_scored'] == 0) & (group['player_out'] == 1)).astype(int)
        group['rolling_ducks'] = group['ducks'].shift().rolling(n3, min_periods=1).sum()
        group['rolling_ducks'] = group['rolling_ducks'] ** 2
        return group

    def calculate_rolling_maidens(self, group, n3):
        """Calculate the rolling sum of maidens over the last n matches."""
        group['rolling_maidens'] = group['maidens'].shift().rolling(n3, min_periods=1).sum()
        return group

    def calculate_alpha_batsmen_score_for_odis(self, group, n3):
        """Calculate the α_batsmen_score tailored for Dream11 point prediction in ODIs, including strike rotation percentage and refined metrics."""
        group = group.sort_values('start_date')

        # Rolling averages for the last n matches
        group['avg_runs_scored'] = group['runs_scored'].shift().rolling(n3, min_periods=1).mean()
        group['avg_strike_rate'] = group['strike_rate_n1'].shift().rolling(n3, min_periods=1).mean()
        group['avg_half_centuries'] = group['half_centuries_cumsum'].shift().rolling(n3, min_periods=1).sum()
        group['avg_centuries'] = group['centuries_cumsum'].shift().rolling(n3, min_periods=1).sum()  # Cumulative centuries
        group['avg_rolling_ducks'] = group['rolling_ducks'].shift().rolling(n3, min_periods=1).sum()

        # Shift runs and boundary runs to exclude the current match and calculate singles scored
        group['boundary_runs'] = (group['fours_scored'].shift() * 4) + (group['sixes_scored'].shift() * 6)
        group['singles_scored'] = group['runs_scored'].shift() - group['boundary_runs']

        # Strike Rotation Percentage: (singles / balls faced) * 100
        group['strike_rotation_percentage'] = (group['singles_scored'] / group['balls_faced'].shift() - group['fours_scored'].shift() - group['sixes_scored'].shift()) * 100
        group['avg_strike_rotation_percentage'] = group['strike_rotation_percentage'].shift().rolling(n3, min_periods=1).mean()

        # Replace NaN values with 0 before calculating the α_batsmen_score
        group.fillna(0, inplace=True)
        
        # Calculate the α_batsmen_score for ODIs (without average fours and sixes, and including strike rotation)
        group['α_batsmen_score'] = (
            0.25 * group['avg_runs_scored'] +          # Runs scored (core contribution)
            0.10 * group['avg_strike_rate'] +         # Moderate weight on strike rate
            0.20 * group['avg_centuries'] +           # Significant weight on centuries
            0.15 * group['avg_half_centuries'] +     # Rewards for scoring half-centuries (key milestone)
            1.5 * group['avg_rolling_ducks'] +        # Moderate penalty for ducks
            0.15 * group['avg_strike_rotation_percentage']  # New term for strike rotation percentage
        )

        return group

    def calculate_alpha_bowler_score(self, group, n3):
        """Calculate the α_bowler_score for T20s, focusing on Dream11 fantasy point prediction."""
        group = group.sort_values('start_date')

        # Rolling averages for the last n matches
        group['avg_wickets_taken'] = group['wickets_taken'].shift().rolling(n3, min_periods=1).mean()
        group['avg_bowling_average'] = group['bowling_average_n1'].shift().rolling(n3, min_periods=1).mean()
        group['avg_bowling_strike_rate'] = group['bowling_strike_rate_n1'].shift().rolling(n3, min_periods=1).mean()
        group['avg_economy_rate'] = group['economy_rate_n1'].shift().rolling(n3, min_periods=1).mean()
        group['avg_maidens'] = group['rolling_maidens'].shift().rolling(n3, min_periods=1).sum()

        # Replace NaN values with 0 before calculating the α_bowler_score
        group.fillna(0, inplace=True)

        # Calculate the α_bowler_score for T20, with appropriate weights for Dream11 points
        group['α_bowler_score'] = (
            0.35 * group['avg_wickets_taken'] +           # Wickets taken (core metric for T20 fantasy)
            0.25 * group['avg_bowling_strike_rate'] +     # Strike rate (key for Dream11 in T20s)
            0.20 * group['avg_economy_rate'] +            # Economy rate (penalized in T20s if high)
            0.10 * group['avg_maidens'] -                 # Maidens (rare but valuable in T20)
            0.10 * group['avg_bowling_average']           # Bowling average (penalized if high)
        )
        
        return group

    def assign_rating_score(self, group):
        """
        Assign batsman and bowler ratings based on predefined ranges.
        Parameters:
        - group: DataFrame containing player performance data with 'α_batsmen_score' and 'α_bowler_score'.
        Returns:
        - group: DataFrame with 'batsman_rating' and 'bowler_rating' added.
        """
        # Define batsman rating ranges
        batsman_ranges = {
            (0, 5): 0,
            (5, 15): 4,
            (15, 25): 9,
            (25, 35): 16,
            (35, 45): 25,
            (45, 55): 49,
            (55, float('inf')): 81
        }

        # Define bowler rating ranges
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

        # Helper function to assign rating based on ranges
        def get_rating(score, ranges):
            for (lower, upper), rating in ranges.items():
                if lower <= score < upper:
                    return rating
            return 0  # Default in case no range matches

        # Apply batsman and bowler ratings
        group['batsman_rating'] = group['α_batsmen_score'].apply(lambda x: get_rating(x, batsman_ranges))
        group['bowler_rating'] = group['α_bowler_score'].apply(lambda x: get_rating(x, bowler_ranges))

        return group

    def run_30_to_50(self, group):
        group = group.sort_values('start_date')
        group['cumulative_30s'] = group['runs_scored'].shift().expanding(min_periods=1).apply(self.calculate_30s)
        group['conversion_30_to_50'] = group.apply(lambda x: (x['half_centuries_cumsum'] / x['cumulative_30s']) if x['cumulative_30s'] != 0 else 0, axis=1)
        return group

    def calculate_30s(self, runs_scored):
        """Calculate the total number of 30s."""
        return ((runs_scored >= 30) & (runs_scored < 50)).sum()

    def longtermfeatures(self, group):
        """Calculate long-term career features for batting and bowling."""
        group = group.sort_values('start_date')
        
        # Batting features
        group['longterm_avg_runs'] = group['runs_scored'].shift().expanding().mean()
        group['longterm_var_runs'] = np.sqrt(group['runs_scored'].shift().expanding().var())
        group['longterm_avg_strike_rate'] = (
            (group['runs_scored'].shift().expanding().sum()) /
            (group['balls_faced'].shift().expanding().sum()) * 100
        )

        # Bowling features
        group['longterm_avg_wickets_per_match'] = group['wickets_taken'].shift().expanding().mean()
        group['longterm_var_wickets_per_match'] = np.sqrt(group['wickets_taken'].shift().expanding().var())
        group['longterm_avg_economy_rate'] = (
            (group['runs_conceded'].shift().expanding().sum()) /
            ((group['balls_bowled'].shift().expanding().sum()) / group['balls_per_over'].iloc[0])
        )
        group['longterm_total_matches_of_type'] = group.groupby('match_type').cumcount()
        return group

    def order_seen(self, group):
        group['order_seen_mode'] = group['order_seen'].shift().expanding().apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
        return group

    def rolling_dot_balls_features(self, group, n1, n2, n3):
        """
        Calculate bowling averages, economy rates, strike rates, updated CBR, 
        and fielding points using a rolling window.
        """
        group = group.sort_values('start_date')

        def calculate_rolling_metrics(group, n, min_periods, name):
            balls = group['balls_bowled'].shift().rolling(n, min_periods=min_periods).sum()
            group[name] = (group['dot_balls_as_bowler'].shift().rolling(n, min_periods=min_periods).sum() / balls) * 100

            return group

        # Calculate metrics for each window size
        group = calculate_rolling_metrics(group, n1, 1, 'dot_ball_percentage_n1')
        group = calculate_rolling_metrics(group, n2, 3, 'dot_ball_percentage_n2')
        group = calculate_rolling_metrics(group, n3, 5, 'dot_ball_percentage_n3')

        return group

    def longtermfeatures_dot_balls(self, group):
        """Calculate long-term dot ball_percentage"""
        group = group.sort_values('start_date')
        balls = group['balls_bowled'].shift().expanding().sum()
        group['longterm_dot_ball_percentage'] = (group['dot_balls_as_bowler'].shift().expanding().sum() / balls) * 100
        group['dot_ball_percentage'] = (group['dot_balls_as_bowler'].shift().expanding().sum() / group['balls_bowled'].shift().expanding().sum()) * 100
        group['longterm_var_dot_ball_percentage'] = np.sqrt(group['dot_ball_percentage'].shift().expanding().var())

        return group

    def player_features(self,data, n1,n2,n3):
        """Main function to calculate player features using helper functions."""
        cutoff_date = pd.to_datetime('2024-12-06')
        data = self.prepare_data(data, cutoff_date)

        feature_data = []

        for (name, match_type), group in data.groupby(['player_id', 'match_type']):
            group = group.sort_values('start_date')
            group=self.longtermfeatures(group)
            group = self.calculate_rolling_batting_stats_with_context(group, n1, n2, n3)
            group = self.calculate_rolling_bowling_stats_with_ema(group, n1, n2, n3)
            group = self.calculate_centuries_and_half_centuries(group)
            group = self.calculate_additional_stats(group, n3)
            group = self.calculate_rolling_fantasy_score(group)
            group = self.calculate_rolling_ducks(group, n3)
            group = self.calculate_rolling_maidens(group, n3)
            group = self.calculate_alpha_batsmen_score_for_odis(group, n3)
            group = self.calculate_alpha_bowler_score(group, n3)
            group=self.assign_rating_score(group)
            group=self.order_seen(group)
            group=self.run_30_to_50(group)
            group= self.rolling_dot_balls_features(group, n1,n2,n3)
            group=self.longtermfeatures_dot_balls(group)
            feature_data.append(group[['player_id','match_id' ,'match_type', 'start_date',
                            # Batting stats for n1, n2, n3
                            'batting_average_n1', 'strike_rate_n1', 'boundary_percentage_n1',
                            'batting_average_n2', 'strike_rate_n2', 'boundary_percentage_n2',
                            'batting_average_n3', 'strike_rate_n3', 'boundary_percentage_n3',
                            
                            # Cumulative centuries and half-centuries
                            'centuries_cumsum', 'half_centuries_cumsum',
                            
                            # Additional batting stats (avg, half centuries, etc.)
                            'avg_runs_scored', 'avg_strike_rate', 'avg_half_centuries', 'avg_centuries', 
                            'avg_rolling_ducks', 'strike_rotation_percentage', 'avg_strike_rotation_percentage','conversion_30_to_50',
                            
                            # Bowling stats for n1, n2, n3
                            'economy_rate_n1', 'economy_rate_n2', 'economy_rate_n3',
                            'wickets_in_n_matches', 'total_overs_throwed', 
                            'bowling_average_n1', 'bowling_strike_rate_n1', 
                            'bowling_average_n2', 'bowling_strike_rate_n2',
                            'bowling_average_n3', 'bowling_strike_rate_n3',
                            
                            # CBR and CBR2 (bowling performance measure)
                            'CBR', 'CBR2',
                            
                            # Fielding and additional stats
                            'fielding_points', 'four_wicket_hauls_n', 'highest_runs', 'highest_wickets', 
                            'order_seen_mode',
                            
                            # Long-term features for batting and bowling
                            'longterm_avg_runs', 'longterm_var_runs', 'longterm_avg_strike_rate',
                            'longterm_avg_wickets_per_match', 'longterm_var_wickets_per_match',
                            'longterm_avg_economy_rate', 'longterm_total_matches_of_type',
                            
                            # Fantasy score averages
                            'avg_fantasy_score_1','avg_fantasy_score_5','avg_fantasy_score_10',
                            'avg_fantasy_score_15','avg_fantasy_score_20',
                            
                            # Rolling ducks and maidens
                            'rolling_ducks', 'rolling_maidens',
                            
                            # Player's final score
                            'α_batsmen_score', 'α_bowler_score', 'dot_ball_percentage_n1','dot_ball_percentage_n2','dot_ball_percentage_n3','longterm_dot_ball_percentage','dot_ball_percentage','longterm_var_dot_ball_percentage',
                            
                            # Rating for batsman and bowler
                            'batsman_rating', 'bowler_rating']])

        result_df = pd.concat(feature_data)
        result_df = result_df.reset_index(drop=True)  # Reset index and drop old index
        data=data.merge(result_df, on=['player_id', 'match_type','match_id', 'start_date'], how='left')
        data = data.sort_values(by='start_date').reset_index(drop=True)

        return data

    def calculate_match_level_venue_stats(self,df, lower_param=4.5, upper_param=7):

        # Ensure only the first date is used if dates contain multiple entries
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
            # Total number of balls per match type
            balls_per_inning = 120 if group['match_type'].iloc[0] == 'T20' else 300

            # Calculate the row number within each group to determine the number of innings processed
            group = group.reset_index(drop=True)  # Ensure a contiguous index within the group
            group['inning_number'] = group.index

            # Calculate cumulative stats up to (but excluding) the current match
            group['cumulative_runs'] = group['runs_off_bat'].shift(1).fillna(0).cumsum()
            group['cumulative_wickets'] = group[dismissal_cols].sum(axis=1).shift(1).fillna(0).cumsum()

            # Cumulative balls based on the inning number
            group['cumulative_balls'] = 2 * balls_per_inning
            group['overs'] = group['cumulative_balls'] / 6  # Convert balls to overs

            # Calculate ARPO
            group['ARPO_venue'] = group['cumulative_runs'] / group['overs']

            # Derived stats
            #group['Boundary_Percentage_venue'] = (group['cumulative_runs'] / group['cumulative_balls']) * 100
            group['BSR_venue'] = (group['cumulative_runs'] / group['cumulative_balls']) * 100

            # Average First-Innings Score (AFIS)
            group['AFIS_venue'] = group[group['innings'] == 1]['runs_off_bat'].expanding().mean()
            group['ASIS_venue'] = group[group['innings'] == 2]['runs_off_bat'].expanding().mean()
            # Classify pitch type based on ARPO and thresholds
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
            'ARPO_venue': 'last',                     # ARPO as of this match
        #    'Boundary_Percentage_venue': 'last',      # Boundary Percentage
            'ASIS_venue': 'last',
            'BSR_venue': 'last',                      # Batting Strike Rate
            'AFIS_venue': 'last',                     # Average First-Innings Score
            'Pitch_Type': 'last'                      # Pitch classification
        }).reset_index()

        return match_stats

    def calculate_and_merge_selected_stats(self,match, data, lower_param, upper_param, additional_columns=None):
        # Calculate match-level stats
        match_level_stats = self.calculate_match_level_venue_stats(match, lower_param, upper_param)

        # Select only the required columns for merging
        selected_columns = ['match_id', 'ARPO_venue', 'BSR_venue', 'Pitch_Type']
        match_level_stats = match_level_stats[selected_columns]

        # If additional columns are specified, include them
        if additional_columns:
            additional_data = match[['match_id'] + additional_columns].drop_duplicates()
            match_level_stats = pd.merge(match_level_stats, additional_data, on='match_id', how='left')

        # Merge the selected stats and additional columns with the provided dataframe
        merged_df = pd.merge(
            data,
            match_level_stats,
            on=['match_id'],
            how='left'
        )

        return merged_df

    def apply_nationality(self,df):
        countries = [
            'Afghanistan', 'Åland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 
            'Anguilla', 'Antarctica', 'Antigua & Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 
            'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 
            'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia & Herzegovina', 'Botswana', 'Bouvet Island', 
            'Brazil', 'British Indian Ocean Territory', 'British Virgin Islands', 'Brunei', 'Bulgaria', 
            'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Caribbean Netherlands', 
            'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 
            'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo - Brazzaville', 'Congo - Kinshasa', 
            'Cook Islands', 'Costa Rica', 'Côte d’Ivoire', 'Croatia', 'Cuba', 'Curaçao', 'Cyprus', 'Czechia', 
            'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 
            'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Falkland Islands', 
            'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 
            'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 
            'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 
            'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard & McDonald Islands', 'Honduras', 'Hong Kong', 'Hungary', 
            'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 
            'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 
            'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 
            'Luxembourg', 'Macao SAR China', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 
            'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia', 
            'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 
            'Myanmar (Burma)', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 
            'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'North Korea', 'North Macedonia', 
            'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territories', 
            'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn Islands', 'Poland', 
            'Portugal', 'Puerto Rico', 'Qatar', 'Réunion', 'Romania', 'Russia', 'Rwanda', 'Samoa', 
            'San Marino', 'São Tomé & Príncipe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 
            'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 
            'South Africa', 'South Georgia & South Sandwich Islands', 'South Korea', 'South Sudan', 'Spain', 
            'Sri Lanka', 'St. Barthélemy', 'St. Helena', 'St. Kitts & Nevis', 'St. Lucia', 'St. Martin', 
            'St. Pierre & Miquelon', 'St. Vincent & Grenadines', 'Sudan', 'Suriname', 'Svalbard & Jan Mayen', 
            'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 
            'Togo', 'Tokelau', 'Tonga', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 
            'Turks & Caicos Islands', 'Tuvalu', 'U.S. Outlying Islands', 'U.S. Virgin Islands', 'Uganda', 
            'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 
            'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Wallis & Futuna', 'Western Sahara', 'Yemen', 
            'Zambia', 'Zimbabwe'
        ]

        # List of Indian states
        indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
            "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", 
            "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
            "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", 
            "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", 
            "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
        ]

        # Map the players to Indian states if not already mapped
        df['nationality'] = df['player_team'].where(df['player_team'].isin(countries), None)

        # Map players with teams in Indian states to 'India'
        df['nationality'] = df['nationality'].where(
            df['nationality'].notna() | ~df['player_team'].isin(indian_states),
            'India'
        )

        # Propagate the country_new for each player_id to all occurrences of the same player_id
        df['nationality'] = df.groupby('player_id')['nationality'].transform('first')

        return df


    def process_country_and_homeaway(self,df, final_df):
        """
        Processes the data to map city names to country names and determine home/away status.

        Args:
        df (pd.DataFrame): DataFrame containing 'city' and 'match_id' columns.
        final_df (pd.DataFrame): DataFrame containing 'match_id' and 'player_team' columns.

        Returns:
        pd.DataFrame: Updated final_df with 'country_ground' and 'home_away' columns.
        """
        # Initialize geonamescache and create city-to-country mapping
        gc = geonamescache.GeonamesCache()
        cities = gc.get_cities()
        city_to_countrycode = {info['name']: info['countrycode'] for code, info in cities.items()}

        # Function to map city to country
        def get_country(city_name):
            country_code = city_to_countrycode.get(city_name)
            if not country_code:
                return "Unknown Country"
            country = pycountry.countries.get(alpha_2=country_code)
            return country.name if country else "Unknown Country"

        # Map city names in df to country names
        df['country_ground'] = df['city'].apply(get_country)

        # Create a dictionary mapping match_id to country_ground
        match_id_to_country = df.set_index('match_id')['country_ground'].to_dict()

        # Map match_id in final_df to country_ground
        final_df['country_ground'] = final_df['match_id'].map(match_id_to_country)

        # Define the homeaway function
        def homeaway(country_of_player, country_venue):
            if country_venue == 'Unknown Country':
                return 'neutral'
            elif country_of_player == country_venue:
                return 'home'
            else:
                return 'away'

        # Determine home/away status for each row in final_df
        final_df['home_away'] = final_df.apply(
            lambda row: homeaway(row['nationality'], row['country_ground']), axis=1
        )

        return final_df

    def avg_against_opposition(self,final_df):
        final_df['avg_against_opposition'] = (
        final_df.groupby(['player_id', 'opposition_team'])['fantasy_score_total']
        .apply(lambda x: x.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)  # Align back to the original index
        )
        return final_df

    def calculate_matches_played_before(self,df):
        """
        Calculate the total number of matches a player has played in a particular match type before the current match.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame containing match data.
        
        Returns:
        pd.DataFrame: The DataFrame with an additional column 'matches_played_before' indicating the total number of matches played before the current match.
        """
        # Ensure the DataFrame is sorted by player_id, match_type, and start_date
        df = df.sort_values(by=['player_id', 'match_type', 'start_date'])
        
        # Group by player_id and match_type
        df['longterm_total_matches_of_type'] = df.groupby(['player_id', 'match_type']).cumcount()
        
        return df

    def assign_batter_bowler(self,final_df):
        final_df['batter'] = np.where(
            pd.isna(final_df['playing_role']),  # Check if 'playing_role' is null
            0,  # Assign 0 if null
            np.where(
                final_df['playing_role'].str.contains(r'Batter|Wicketkeeper|Batting Allrounder', case=False, na=False),
                1,
                0
            )
        )

        # Handle 'bowler' column
        final_df['bowler'] = np.where(
            pd.isna(final_df['playing_role']),  # Check if 'playing_role' is null
            0,  # Assign 0 if null
            np.where(
                final_df['playing_role'].str.contains(r'Bowler|Bowling Allrounder', case=False, na=False),
                1,
                0
            )
        )
        return final_df

    def calculate_opponent_fantasy_avg(self,final_df):
        # Filter the dataframe to include only rows from the last three years
        three_years_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        recent_df = final_df[final_df['start_date'] >= three_years_ago]

        # Pre-compute average fantasy scores for all batters and bowlers grouped by team
        avg_batting_scores = (
            recent_df[recent_df['batter'] == 1]
            .groupby('opposition_team')['fantasy_score_batting']
            .mean()
            .to_dict()
        )

        avg_bowling_scores = (
            recent_df[recent_df['bowler'] == 1]
            .groupby('opposition_team')['fantasy_score_bowling']
            .mean()
            .to_dict()
        )

        # Initialize new columns
        final_df['opponent_avg_fantasy_batting'] = 0.0
        final_df['opponent_avg_fantasy_bowling'] = 0.0

        # Apply the pre-computed averages for batters and bowlers
        def compute_opponent_avg(row):
            opposition_team = row['opposition_team']
            if row['bowler'] == 1:
                # Fetch pre-computed average fantasy score batting for the opposition team
                row['opponent_avg_fantasy_batting'] = avg_batting_scores.get(opposition_team, 0.0)
            if row['batter'] == 1:
                # Fetch pre-computed average fantasy score bowling for the opposition team
                row['opponent_avg_fantasy_bowling'] = avg_bowling_scores.get(opposition_team, 0.0)
            return row

        # Use apply to calculate averages for all rows
        final_df = final_df.apply(compute_opponent_avg, axis=1)

        return final_df

    def calculate_impact_scores(self,final_df, n2):
        # Sort matches chronologically
        final_df = final_df.sort_values(by='start_date')
        # Initialize impact score columns for each format
        final_df['odi_impact'] = 0.0
        final_df['odi_impact'] = (
            (final_df[f'batting_average_{n2}'] * final_df[f'strike_rate_{n2}'] / 120) * final_df['role_factor']
            + 0.15 * final_df[f'boundary_percentage_{n2}']
            + 0.05 * final_df[f'dot_ball_percentage_{n2}']
        )
        return final_df

    def get_role_factor(self,position):
        if position <= 3:  # Top Order
            return 1.2
        elif position <= 6:  # Middle Order
            return 1.0
        else:  # Lower Order
            return 0.8

    def categorize_bowling_style(self,style):
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

    def target_encode(self,df, column, target, smoothing=1):
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
        # Calculate the global mean of the target
        global_mean = df[target].mean()
        
        # Group by the column to calculate group-specific means and counts
        agg = df.groupby(column)[target].agg(['mean', 'count'])
        
        # Compute the smoothed target mean
        smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - 1) / smoothing))
        agg['smoothed_mean'] = global_mean * (1 - smoothing_factor) + agg['mean'] * smoothing_factor
        
        # Map the smoothed means back to the original column
        encoded_series = df[column].map(agg['smoothed_mean'])
        
        return encoded_series

    def binaryclassification(self,final_df):
        match_id_counts = final_df['match_id'].value_counts()

        valid_match_ids = match_id_counts[match_id_counts == 22].index

        # Filter the original DataFrame
        filtered_final_df = final_df[final_df['match_id'].isin(valid_match_ids)]

        # Initialize the 'selected' column with 0
        filtered_final_df['selected'] = 0

        # For each 'match_id', sort by 'fantasy_score_total' and mark the top 11 rows
        for match_id in valid_match_ids:
            match_rows = filtered_final_df[filtered_final_df['match_id'] == match_id]
            top_11_indices = match_rows.sort_values(by='fantasy_score_total', ascending=False).head(11).index
            filtered_final_df.loc[top_11_indices, 'selected'] = 1
        
        # Reset index if needed
        filtered_final_df = filtered_final_df.reset_index(drop=True)
        return filtered_final_df
    
    def one_hot_encode(self,X, column_name):
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
    def onehotencode(self,df):
        df['Pitch_Type'].fillna('Neutral', inplace=True)
        df = self.one_hot_encode(df, 'home_away')
        df=self.one_hot_encode(df,'Pitch_Type')
        df=self.one_hot_encode(df,'gender')
        return df

    def generate_features_ODI(self):

        df = pd.read_csv(file_path_mw_pw, index_col=False)
        df = df[(df['match_type'] == 'ODM') | (df['match_type'] == 'ODI')]
        match = pd.read_csv(file_path_mw_overall, index_col=False)

        df['player_id'] = df['player_id'].astype(str)
        df['match_id'] = df['match_id'].astype(str)
        df['start_date'] = pd.to_datetime(df['start_date']).astype(str)

        df = df.sort_values(by='start_date').reset_index(drop=True)

        df = calculate_fantasy_scores(df)
        match['match_id'] = match['match_id'].astype(str)
        df = self.calculate_and_merge_selected_stats(match, df, 3, 6)
        df = self.apply_nationality(df)
        df = self.process_country_and_homeaway(match, df)
        df = self.player_features(df, 3, 7, 12)
        df = self.avg_against_opposition(df)
        df = self.assign_batter_bowler(df)
        df = self.calculate_matches_played_before(df)
        df = df.sort_values(by=['start_date']).reset_index(drop=True)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = self.calculate_opponent_fantasy_avg(df)
        df['role_factor'] = df['order_seen_mode'].apply(self.get_role_factor)
        df = self.calculate_impact_scores(df, "n2")
        df['bowling_style'] = df['bowling_style'].apply(self.categorize_bowling_style)
        df['bowling_style'] = self.target_encode(df, 'bowling_style', 'fantasy_score_total')
        df = self.binaryclassification(df)
        df = self.onehotencode(df)

        out_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src", "data", "processed", "final_training_file_odi.csv"))
        df.to_csv(out_path, index=False)

def main_odi():
    obj = FeatureGeneratorODI()
    obj.generate_features_ODI()


def main_feature_generation(): 
    main_odi()
    main_t20()
    main_test()

 




