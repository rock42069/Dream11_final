import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths


def process_matches():

    pred_path = os.path.abspath(os.path.join(current_dir, ".." , "..","src", "data", "processed", "predictions_test.csv"))
    data_path = os.path.abspath(os.path.join(current_dir, ".." , "..","src", "data", "interim", "mw_pw_profiles.csv"))
    df_pred = pd.read_csv(pred_path, index_col=False)
    df_data = pd.read_csv(data_path, index_col=False)

    df_pred['match_id'] = df_pred['match_id'].astype(str)
    df_data['match_id'] = df_data['match_id'].astype(str)

    df_merged = df_pred.merge(df_data[['match_id', 'start_date', 'player_team', 'player_id', 'full_name']], on=['match_id', 'player_id'], how='left')
    df_output = df_merged[['match_id', 'start_date', 'player_team', 'stacked_model_predicted_score', 'fantasy_score_total', 'player_id', 'full_name']]

    result_list = []

    for match_id in df_output['match_id'].unique():
        match_df = df_output[df_output['match_id'] == match_id]
        Match_date = match_df['start_date'].iloc[0]
        Team_1 = match_df['player_team'].unique()[0]
        Team_2 = match_df['player_team'].unique()[1]

        # Select top 11 players based on predicted score
        top_predicted = match_df.sort_values(by="stacked_model_predicted_score", ascending=False).head(11)
        predicted_players = {
            f"Predicted Player {i+1}": top_predicted.iloc[i]["full_name"] for i in range(11)
        }
        predicted_scores = {
            f"Predicted Player {i+1} Points": top_predicted.iloc[i]["stacked_model_predicted_score"] for i in range(11)
        }

        # Select top 11 players based on fantasy score
        top_fantasy = match_df.sort_values(by="fantasy_score_total", ascending=False).head(11)
        fantasy_players = {
            f"Dream team player {i+1}": top_fantasy.iloc[i]["full_name"] for i in range(11)
        }
        fantasy_scores = {
            f"Dream team player {i+1} Points": top_fantasy.iloc[i]["fantasy_score_total"] for i in range(11)
        }

        # Combine results
        result = {}
        for i in range(11):
            result[f"Predicted Player {i+1}"] = predicted_players[f"Predicted Player {i+1}"]
            result[f"Predicted Player {i+1} Points"] = predicted_scores[f"Predicted Player {i+1} Points"]
        for i in range(11):
            result[f"Dream team player {i+1}"] = fantasy_players[f"Dream team player {i+1}"]
            result[f"Dream team player {i+1} Points"] = fantasy_scores[f"Dream team player {i+1} Points"]

        # Create result DataFrame
        result_df = pd.DataFrame([result])
        result_df.insert(0, 'Match Date', Match_date)
        result_df.insert(1, 'Team 1', Team_1)
        result_df.insert(2, 'Team 2', Team_2)

        # Sum the Dream team player points
        result_df['Total Dream Team Points'] = sum(top_fantasy['fantasy_score_total'])
        result_df['Total Predicted Players Fantasy Points'] = sum(top_predicted['fantasy_score_total'])
        result_df['Total Points MAE'] = abs(result_df['Total Dream Team Points'] - result_df['Total Predicted Players Fantasy Points'])

        result_list.append(result_df)

    final_result_df = pd.concat(result_list, ignore_index=True)
    final_path = os.path.abspath(os.path.join(current_dir , ".." , ".." , "src","data" ,"processed" ,"final_output.csv"))
    final_result_df.to_csv( final_path,index=False)
    return final_result_df

