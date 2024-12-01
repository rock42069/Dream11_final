import json
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import requests
import csv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

this_file_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

json_dir = this_file_dir + "../data/raw/cricksheet/json/"
csv_dir = this_file_dir + "../data/raw/cricksheet/csv/"
counter = 0
counter_lock = Lock()

#################################
# FILE FIND: data_download.py
#################################
def execute_scraper():
    json_url = "https://cricsheet.org/downloads/all_json.zip"
    csv_url = "https://cricsheet.org/downloads/all_csv2.zip"
    people_csv_url = "https://cricsheet.org/register/people.csv"

    target_json_dir = this_file_dir + '../data/raw/cricksheet/json/'
    target_csv_dir = this_file_dir + '../data/raw/cricksheet/csv/'
    target_people_csv_path = this_file_dir + "../data/raw/cricksheet/people.csv"

    json_zip_file = "all_json.zip"
    csv_zip_file = "all_csv2.zip"

    # make sure all the directories exist, if not create them using the os module
    if not os.path.exists(target_json_dir):
        os.makedirs(target_json_dir)
    if not os.path.exists(target_csv_dir):
        os.makedirs(target_csv_dir)
    
    response = requests.get(people_csv_url)
    with open(target_people_csv_path, 'w') as file:
        file.write(response.text)

    # download the zip files
    os.system(f"curl {json_url} -O {json_zip_file}")
    os.system(f"curl {csv_url} -O {csv_zip_file}")
    #os.system(f"curl {people_csv_url} -O {target_people_csv_path}")

    # unzip the files
    os.system(f"unzip {json_zip_file} -d {target_json_dir}")
    os.system(f"unzip {csv_zip_file} -d {target_csv_dir}")

    # delete the zip files
    os.system(f"rm {json_zip_file}")
    os.system(f"rm {csv_zip_file}")


    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # scraper_path = os.path.join(script_dir, 'scraper.sh')

    # os.system(f'chmod +x {scraper_path}')    
    # subprocess.call(['sh', scraper_path, '-d', this_file_dir])

#################################
# FILE FIND: json_generator.py
#################################
dir_with_cricsheets_json = json_dir
stored_dir = dir_with_cricsheets_json

output_csv_json_generator = this_file_dir + "../data/interim/total_data.csv" # json -> csv with 2 rows per player per match (1 per inning)

match_types_unique = set()


match_attributes = [
    "match_id",
    "gender",
    "balls_per_over",
    "date",
    "series_name",
    "match_type"
]

batsman_attributes = [ #now contains all the attributes
    "player_id",
    "runs_scored",
    "player_out", 
    "balls_faced",
    "fours_scored",
    "sixes_scored",
    "catches_taken",
    "run_out_direct",
    "run_out_throw",
    "stumpings_done",
    "out_kind",
    "dot_balls_as_batsman",
    "order_seen",
    "balls_bowled", 
    "runs_conceded",
    "wickets_taken",
    "bowled_done",
    "lbw_done",
    "maidens",
    "dot_balls_as_bowler",
    "player_team",
    "opposition_team"
]

bowler_attributes = [
    "catches_taken",
    "run_out_direct",
    "run_out_throw",
    "balls_bowled", 
    "runs_conceded",
    "wickets_taken",
    "catches_taken",
    "bowled_done",
    "lbw_done",
    "maidens",
    "dot_balls_as_bowler"
]

def import_data(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
            
    
    info_data = json_data.get("info", {})
    info_data["match_id"] = file_path.split("/")[-1].split(".")[0]
    innings_data = json_data.get("innings", [])
    
    
    return info_data, innings_data

def fix_dates(dates):
    ret = ""
    if len(dates) > 1:
        ret = dates[0] + " - " + dates[1]
    else:
        ret = dates[0]
    return ret

def parse_info_data(info_data):
    dates = info_data["dates"]
    dates = fix_dates(dates)
    info_data["date"] = dates
    try:
        info_data["series_name"] = info_data["event"]["name"].replace(",", " ")
    except:
        info_data["series_name"] = "None"
    ret = {}

    match_type = info_data["match_type"]

    if match_type not in match_types_unique:
        match_types_unique.add(match_type)

    for attribute in match_attributes:
        ret[attribute] = info_data.get(attribute, None)
    return ret

def get_players_data_dict(players_in_match, player_ids, info_dict): # returns a dictionary with all the players in the match and their attributes
    total_data = {}
    
    for player in players_in_match:
        total_data[player] = {}
        player_id = player_ids[player]
        for attribute in batsman_attributes:
            total_data[player][attribute] = 0
        for attribute in bowler_attributes:
            total_data[player][attribute] = 0
        total_data[player]["player_id"] = player_id
    
    team_players_mapping = info_dict["players"]
    team_1 = list(team_players_mapping.keys())[0]
    team_2 = list(team_players_mapping.keys())[1]

    for player in team_players_mapping[team_1]:
        total_data[player]["player_team"] = team_1
        total_data[player]["opposition_team"] = team_2
    for player in team_players_mapping[team_2]:
        total_data[player]["player_team"] = team_2
        total_data[player]["opposition_team"] = team_1

    return total_data

def get_overs(session):
    try:
        overs = session["overs"]
    except:
        overs = []
    return overs

def is_wicket(ball):
    try:
        wicket = ball["wickets"]
        return True
    except:
        return False

def get_wicket_data(wicket):
    ret = {}
    wicket = wicket[0]
    ret["player_out"] = wicket["player_out"]
    ret["kind"] = wicket["kind"]
    try:
        ret["fielders"] = wicket["fielders"]
    except:
        ret["fielders"] = None
    return ret

def split_data(total_data):
    batsmen = []
    bowlers = []

    for player in total_data:
        batsmen.append(player)
        bowlers.append(player)

    batsmen_data = {}
    bowlers_data = {}

    for player in batsmen:
        batsmen_data[player] = {}
        for attribute in batsman_attributes:
            batsmen_data[player][attribute] = total_data[player][attribute]
    
    for player in bowlers:
        bowlers_data[player] = {}
        for attribute in bowler_attributes:
            bowlers_data[player][attribute] = total_data[player][attribute]
    return batsmen_data, bowlers_data


def export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order):
    if not os.path.exists(output_csv_json_generator):
        with open(output_csv_json_generator, 'w') as f:
            for i in match_attributes:
                f.write(i + ',')
            f.write("name,")
            for i in range(len(batsman_attributes)-1):
                f.write(batsman_attributes[i] + ',')
            f.write(batsman_attributes[-1] + '\n')

    for batsman in batsmen:
        try:
            batsmen[batsman]["order_seen"] = batsman_order.index(batsman) + 1
        except:
            pass
        
    with open(output_csv_json_generator, 'a') as f:
        for player in bowlers:
            for i in match_attributes_parsed:
                f.write(str(match_attributes_parsed[i]) + ',')
            f.write(player + ',')
            for i in range(len(batsman_attributes[:-1])):
                try:
                    to_write = batsmen[player][batsman_attributes[i]]
                except:
                    to_write = 0
                f.write(str(to_write) + ',')
            f.write(batsmen[player][batsman_attributes[-1]] + '\n')


def parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, match_info):
    for session in innings_data:
        total_data = get_players_data_dict(players_in_match, player_ids, match_info)
        batsman_order = []
        batsman_data = {}
        bowler_data = {}
        num_seen = 1

        overs = get_overs(session)
        for over in overs: # levels "over" and "deliveries"
            over_number = over["over"]
            over_ball_list = over["deliveries"]
            runs_in_over = 0
            for ball in over_ball_list: # ball by ball
                batsman = ball["batter"]
                bowler = ball["bowler"]
                non_striker = ball["non_striker"]

                if batsman not in batsman_order:
                    total_data[batsman]["order_seen"] = num_seen
                    num_seen += 1
                    batsman_order.append(batsman)

                runs_scored = ball["runs"]["batter"]
                extras = ball["runs"]["extras"]
                runs = runs_scored + extras
                runs_in_over += runs

                if is_wicket(ball):
                    wicket_data = get_wicket_data(ball["wickets"])

                    kind = wicket_data["kind"]

                    total_data[bowler]["wickets_taken"] += 1 # blindly adding, need to remove on kind basis

                    # CATCH
                    if kind == "caught":
                        for fielder in wicket_data["fielders"]:
                            try:
                                fielder = fielder["name"]
                            except:
                                fielder = None
                            try:
                                total_data[fielder]["catches_taken"] += 1
                            except:
                                pass
                    
                    # RUN OUT
                    if kind == "run out":
                        total_data[bowler]["wickets_taken"] -= 1
                        fielders = wicket_data["fielders"]
                        fielders = fielders if fielders != None else []

                        if fielders == []: # broken dataset failsafe
                            pass
                        else:
                            if len(fielders) >= 2:
                                for fielder in fielders:
                                    try:
                                        total_data[fielder["name"]]["run_out_throw"] += 1
                                    except:
                                        pass
                            else:
                                try:
                                    total_data[fielders[0]["name"]]["run_out_direct"] += 1
                                except:
                                    pass
                    

                    # STUMPING
                    if kind == "stumped":
                        total_data[bowler]["wickets_taken"] -= 1
                        fielders = wicket_data["fielders"]
                        try: # broken dataset failsafe
                            total_data[fielders[0]["name"]]["stumpings_done"] += 1
                        except:
                            pass

                    if kind == "bowled":
                        total_data[bowler]["bowled_done"] += 1
                    if kind == "lbw":
                        total_data[bowler]["lbw_done"] += 1
                    total_data[batsman]["out_kind"] = kind
                    total_data[batsman]["player_out"] = 1
                
                    total_data[batsman]["player_out"] = 1
                if not runs_scored:
                    total_data[bowler]["dot_balls_as_bowler"] += 1
                    total_data[batsman]["dot_balls_as_batsman"] += 1

                total_data[bowler]["runs_conceded"] += runs
                total_data[bowler]["balls_bowled"] += 1
                total_data[batsman]["runs_scored"] += runs_scored
                total_data[batsman]["balls_faced"] += 1
                if runs_scored == 4:
                    total_data[batsman]["fours_scored"] += 1
                if runs_scored == 6:
                    total_data[batsman]["sixes_scored"] += 1
            if runs_in_over == 0:
                total_data[bowler]["maidens"] += 1
        batsmen, bowlers = split_data(total_data)
        export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order)
    return batsman_attributes, bowler_attributes

def get_players(info_data):
    players = []
    for i in list(info_data.get("players", {}).values()):
        for player in i:
            players.append(player)
    return players

def generate(file_path):
    info_data, innings_data = import_data(file_path)

    player_ids = info_data["registry"]["people"]

    team_split = info_data["players"]
    players_in_match = get_players(info_data)

    match_attributes_parsed = parse_info_data(info_data)
    parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, info_data)


def json_generator():
    ignore_files = [".", "..", ".DS_Store", "README.txt"]

    total_files = len(os.listdir(stored_dir))
    files_done = 0
    for file in os.listdir(stored_dir):
        files_done += 1
        print(f"Processing file {files_done}/{total_files}")
        print("File: ", file)
        if file not in ignore_files:
            generate(stored_dir + file)
    return 1

#################################
# FILE FIND: mw_overall.py
#################################
output_path_mw_overall = this_file_dir + "../data/interim/mw_overall.csv"

def matchwise_data_generator():
    json_files = sorted(os.listdir(json_dir))
    final_df = []

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        match_key = os.path.splitext(json_file)[0]
        csv_path = os.path.join(csv_dir, f"{match_key}.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping {json_file} - Corresponding CSV file not found")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        teams = data["info"].get("teams", [])
        if len(teams) != 2:
            print(f"Skipping {json_file} - Teams information missing or invalid.")
            continue

        team_players = {team: ", ".join(data["info"].get("players", {}).get(team, [])) for team in teams}

        json_row = {  
            "match_id": match_key,
            "balls_per_over": data["info"].get("balls_per_over", None),
            "city": data["info"].get("city", None),
            "dates": ", ".join(data["info"].get("dates", [])) if data["info"].get("dates") else None,
            "event_name": data["info"].get("event", {}).get("name", None),
            "match_number": data["info"].get("event", {}).get("match_number", None),
            "gender": data["info"].get("gender", None),
            "match_type": data["info"].get("match_type", None),
            "match_type_number": data["info"].get("match_type_number", None),
            "match_referees": ", ".join(data["info"].get("officials", {}).get("match_referees", [])),
            "tv_umpires": ", ".join(data["info"].get("officials", {}).get("tv_umpires", [])),
            "umpires": ", ".join(data["info"].get("officials", {}).get("umpires", [])),
            "result": data["info"].get("outcome", {}).get("result", None),
            "player_of_match": ", ".join(data["info"].get("player_of_match", [])),
            "season": data["info"].get("season", None),
            "team_type": data["info"].get("team_type", None),
            "teams": ", ".join(teams),
            "toss_decision": data["info"].get("toss", {}).get("decision", None),
            "toss_winner": data["info"].get("toss", {}).get("winner", None),
            "venue": data["info"].get("venue", None),
            "winner": (
                "draw" if data["info"].get("outcome", {}).get("result", "").lower() == "draw"
                else data["info"].get("outcome", {}).get("winner", None)
            )
        }
        json_df = pd.DataFrame([json_row])

        match_data = pd.read_csv(csv_path)
        aggregated = match_data.groupby(['match_id', 'innings', 'batting_team']).agg({
            'runs_off_bat': 'sum',
            'extras': 'sum',
            'wides': 'sum',
            'noballs': 'sum',
            'byes': 'sum',
            'legbyes': 'sum',
            'penalty': 'sum',
            'player_dismissed': 'count',
        }).reset_index()

        wicket_counts = match_data.pivot_table(
            index=['match_id', 'innings', 'batting_team'], 
            columns='wicket_type', 
            aggfunc='size', 
            fill_value=0
        ).reset_index()
        aggregated = pd.merge(aggregated, wicket_counts, on=['match_id', 'innings', 'batting_team'], how='left')

        json_df['match_id'] = json_df['match_id'].astype(str)
        aggregated['match_id'] = aggregated['match_id'].astype(str)
        combined = aggregated.merge(json_df, on="match_id", how="left")
        combined['players'] = combined['batting_team'].map(team_players).fillna("Unknown")

        final_df.append(combined)

    final_df = pd.concat(final_df, ignore_index=True)
    final_df.to_csv(output_path_mw_overall, index=False)
    print(f"Processing completed. Data saved to '{output_path_mw_overall}'.")

#################################
# FILE FIND: adding_names.py
#################################
def get_player_details(cricinfo_id, total_players):
    global counter  # Declare the global counter
    url = f"https://www.espncricinfo.com/cricketers/player-{cricinfo_id}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize variables
        full_name = batting_style = bowling_style = playing_role = None
        teams = []
        # Extract full name
        full_name_section = soup.find('div', class_="ds-col-span-2 lg:ds-col-span-1")
        if full_name_section:
            name_label = full_name_section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if name_label and name_label.text == "Full Name":
                full_name = full_name_section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()

        # Extract other details
        info_sections = soup.find_all('div')
        for section in info_sections:
            label = section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if label:
                if label.text == "Batting Style":
                    batting_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Bowling Style":
                    bowling_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Playing Role":
                    playing_role = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()

        teams_section = soup.find('div', class_="ds-grid lg:ds-grid-cols-3 ds-grid-cols-2 ds-gap-y-4")
        if teams_section:
            team_links = teams_section.find_all('a', class_="ds-flex ds-items-center ds-space-x-4")
            for team_link in team_links:
                title = team_link.get('title', '')
                team_name = title.split("'s ", 1)[1].strip()  # Get the part after "'s "
                if team_name.endswith(" team profile"):
                    team_name = team_name[:-13]
                if team_name:
                    teams.append(team_name)
        # Update progress counter safely
        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed.")

        return cricinfo_id, full_name, batting_style, bowling_style, playing_role, teams

    else:
        # Update progress counter for failed requests
        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed (Failed).")

        return cricinfo_id, None, None, None, None, []


def run_scraper_parallel(data, max_workers):
    total_players = len(data)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(get_player_details, row['key_cricinfo'], total_players): row['key_cricinfo']
            for _, row in data.iterrows()
        }

        for future in as_completed(future_to_id):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")

    return results


def aggregate_player_stats(df):
    # Select numerical columns
    num_columns = df.select_dtypes(include='number').columns

    # Group by 'player' and 'match' columns and sum the numerical columns
    aggregated_df = df.groupby(['player_id', 'match_id'], as_index=False)[num_columns].sum()

    # For string columns (if any), use first (or other aggregation method)
    for col in df.select_dtypes(exclude='number').columns:
        aggregated_df[col] = df.groupby(['player_id', 'match_id'])[col].first().values

    return aggregated_df


def adding_names():
    global counter  # Reset counter for a new run
    counter = 0

    people_csv_adding_names = pd.read_csv(this_file_dir + '../data/raw/cricksheet/people.csv')
    weather_csv_adding_names = pd.read_csv(this_file_dir + '../data/interim/total_data.csv') 
    final_csv_path = this_file_dir + "../data/interim/mw_pw.csv"
    global total_players 
    total_players = len(people_csv_adding_names)
    # scraped_data = run_scraper_parallel(people_csv_adding_names, max_workers=30)  # Adjust max_workers based on your machine’s capability

# Convert results to a DataFrame
    # scraped_df = pd.DataFrame(scraped_data, columns=["key_cricinfo", "full_name", "batting_style", "bowling_style", "playing_role", "teams"])
    # final_data = people_csv_adding_names.merge(scraped_df, on='key_cricinfo', how='left')



    final_df = aggregate_player_stats(weather_csv_adding_names)
    final_df.to_csv(final_csv_path, index=False)

def aggregate():
    total_data = pd.read_csv(this_file_dir + "../data/interim/total_data.csv")
    total_data["match_id"] = total_data["match_id"].astype(str)
    total_data["player_id"] = total_data["player_id"].astype(str)


    agg_dict = {
        'gender': 'first',
        'balls_per_over': 'first',
        'start_date': 'first',
        'series_name': 'first',
        'match_type': 'first',
        'name': 'first',
        'runs_scored': 'sum',
        'player_out': 'sum',
        'balls_faced': 'sum',
        'fours_scored': 'sum',
        'sixes_scored': 'sum',
        'catches_taken': 'sum',
        'run_out_direct': 'sum',
        'run_out_throw': 'sum',
        'stumpings_done': 'sum',
        'out_kind': 'first',
        'dot_balls_as_batsman': 'sum',
        'order_seen': 'first',
        'balls_bowled': 'sum',
        'runs_conceded': 'sum',
        'wickets_taken': 'sum',
        'bowled_done': 'sum',
        'lbw_done': 'sum',
        'maidens': 'sum',
        'dot_balls_as_bowler': 'sum',
        'player_team': 'first',
        'opposition_team': 'first'
    }

    result = total_data.groupby(['player_id', 'match_id']).agg(agg_dict).reset_index()
    result.to_csv(this_file_dir + '../data/interim/mw_pw.csv')



def rename_date():
    total_data_path = this_file_dir + "../data/interim/total_data.csv"
    df = pd.read_csv(total_data_path, index_col= False)
    # split date column and take first element by " - "
    df['date'] = df['date'].str.split(" - ").str[0]
    #rename date column
    df.rename(columns={'date': 'start_date'}, inplace=True)
    os.remove(total_data_path)
    df.to_csv(total_data_path)


def get_player_details(cricinfo_id, total_players):
    global counter
    url = f"https://www.espncricinfo.com/cricketers/player-{cricinfo_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize variables
        full_name = batting_style = bowling_style = playing_role = None
        teams = []
        
        # Extract full name
        full_name_section = soup.find('div', class_="ds-col-span-2 lg:ds-col-span-1")
        if full_name_section:
            name_label = full_name_section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if name_label and name_label.text == "Full Name":
                full_name = full_name_section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
        
        # Extract other details
        info_sections = soup.find_all('div')
        for section in info_sections:
            label = section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if label:
                if label.text == "Batting Style":
                    batting_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Bowling Style":
                    bowling_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Playing Role":
                    playing_role = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
        
        teams_section = soup.find('div', class_="ds-grid lg:ds-grid-cols-3 ds-grid-cols-2 ds-gap-y-4")
        if teams_section:
            team_links = teams_section.find_all('a', class_="ds-flex ds-items-center ds-space-x-4")
            for team_link in team_links:
                title = team_link.get('title', '')
                team_name = title.split("'s ", 1)[1].strip()  # Get the part after "'s "
                if team_name.endswith(" team profile"):
                    team_name = team_name[:-13]
                if team_name:
                    teams.append(team_name)


        # Update progress
        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed.")           
        return cricinfo_id, full_name, batting_style, bowling_style, playing_role, teams
    else:
        return cricinfo_id, None, None, None, None, []

def run_scraper_parallel(data, total_players, max_workers):
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(get_player_details, row['key_cricinfo'], total_players): row['key_cricinfo']
            for _, row in data.iterrows()
        }
        
        for future in as_completed(future_to_id):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")
    
    return results

def adding_names():
    global counter

    # Load data
    data = pd.read_csv(this_file_dir + '../data/raw/cricksheet/people.csv')
    total_players = len(data)  # Total number of players
    counter = 0  # Reset counter

    # Run scraper and collect results
    scraped_data = run_scraper_parallel(data, total_players, max_workers=300)

    # Convert results to a DataFrame
    scraped_df = pd.DataFrame(scraped_data, columns=['key_cricinfo', 'full_name', 'batting_style', 'bowling_style', 'playing_role', 'teams'])

    # Merge with the original data to include the scraped fields
    data = data.merge(scraped_df, on='key_cricinfo', how='left')

    # Rename columns in df2 to match df1
    data = data.rename(columns={"identifier": "player_id"})

    # Merge with interim data
    input_data = pd.read_csv(this_file_dir + '../data/interim/mw_pw.csv')
    final_data = input_data.merge(data, on='player_id', how='left')

    # Save the updated data to a new CSV file
    final_data.to_csv(this_file_dir + '../data/interim/mw_pw_profiles.csv', index=False)

    print("Player data updated successfully with parallel scraping.")

def download_and_preprocess():
    
    print("Running execute_scraper()")
    execute_scraper()
    
    print("Running json_generator()")
    json_generator()
    
    rename_date()
    
    print("Running matchwise_data_generator()")
    matchwise_data_generator() # mw_overall.py

    print("Running aggregate()")
    aggregate()
    
    print("Running adding_names()")
    adding_names()


download_and_preprocess()
# adding_names()
# json_generator()
# adding_names()
# aggregate()