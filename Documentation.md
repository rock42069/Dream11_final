# data_download.py
Master function to download and populate `/data/raw/` with the cricsheet data, as well as `/data/interim/` with the interim csv datasheets generated. 

#### `execute_scraper()`
**Description:** The `execute_scraper` function automates the process of downloading, extracting, and organizing cricket match data from Cricsheet into a structured directory. It handles both the available JSON and CSV data formats, as well as downloading the separate CSV file containing player registry.  

The scraper can be tested in the `/src/scraper/cricsheet.py` file

**Input:** This function takes no arguments.  
**Output:** This function creates the following structure and files in the local filesystem:
- A directory containing unzipped JSON files from Cricsheet.  
- A directory containing unzipped CSV files from Cricsheet.  
- A CSV file for player metadata

#### `export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order)`
**Purpose:** Appends match data for all players to the CSV file.  
**Inputs:**  
- `batsmen`: Batting stats dictionary.  
- `bowlers`: Bowling stats dictionary.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `batsman_order`: Batting order list.  

**Output:** Saves the current dataframe/JSON to the appropriate output file path.

#### `parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, match_info)`
**Purpose:** Processes innings data to compute player stats and then exports to CSV.  
**Inputs:**  
- `innings_data`: List of innings details.  
- `players_in_match`: List of player names in the match.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `player_ids`: Player ID mappings.  
- `match_info`: Match metadata.  

**Output:** None

#### `json_generator()`
**Purpose:** Iterates through all JSON files in the directory and processes them.  
**Input:** None  
**Output:** Returns 1 on successful completion.

#### `rename_date()`
**Purpose:** The `rename_date` function processes a CSV file containing cricket match data, modifies the 'date' column, and renames it to 'start_date'. The function ensures that if the match spans multiple days, only the first day is retained in the column. Finally, the function saves the updated data back to the CSV file.

**Inputs:**  
No direct inputs to the function. The function operates on the CSV file located at: `../data/interim/total_data.csv`

**Outputs:**  
Updates the `total_data.csv` file by:
- Splitting the date column based on the delimiter " - " and keeping only the first date.
- Renaming the column `date` to `start_date`.

#### `matchwise_data_generator()`
**Description:** The `matchwise_data_generator` function processes cricket match data by iterating through JSON files and corresponding CSV files, aggregating match statistics, and merging the data into a unified dataframe.

**Inputs:**  
This function does not take direct arguments. It operates on files located in directories specified by the global variables `json_dir` and `csv_dir`.

**Outputs:**  
A combined dataframe containing match data, which is saved as a CSV file at the path defined by `output_path_mw_overall`.

#### `adding_names()`
**Description:** The `adding_names` function aggregates player statistics from match data and incorporates player details such as full name, batting style, bowling style, playing role, and team information. It processes data from multiple CSV files, performs statistical aggregation, and saves the final combined data to a new CSV file.
The scraper can be tested in the `/src/scrapers/cricinfo_scraper.py` file

**Inputs:**  
This function does not take direct arguments. It operates on files located in the following paths:
- `/data/raw/cricksheet/people.csv` (player information).
- `/data/interim/total_data.csv` (match-level data).

**Outputs:**  
A CSV file located at the required output path, which contains the aggregated player statistics along with additional player details (e.g., full name, teams).


# feature_generation.py

#### `rolling_dot_balls_features()`
**Description:** The `rolling_dot_balls_features` function calculates the dot ball percentage for a bowler using rolling windows of different sizes (3, 7, and 12 matches). It computes the metrics by applying a rolling sum to the bowled and dot balls, sorted by match date.

**Inputs:**  
- `group`: A DataFrame containing bowler statistics for a specific player, with columns like `start_date`, `balls_bowled`, and `dot_balls_as_bowler`.  
- `n1`: The window size (in number of matches) for the first dot ball percentage calculation (default is 3).  
- `n2`: The window size (in number of matches) for the second dot ball percentage calculation (default is 7).  
- `n3`: The window size (in number of matches) for the third dot ball percentage calculation (default is 12).  

**Outputs:**  
A DataFrame with additional columns representing the dot ball percentage for the respective rolling window sizes:
- Dot ball percentage for the rolling window of size `n1`.
- Dot ball percentage for the rolling window of size `n2`.
- Dot ball percentage for the rolling window of size `n3`.

#### `longtermfeatures_dot_balls()`
**Description:** The `longtermfeatures_dot_balls` function calculates the long-term dot ball percentage for a bowler using an expanding window. It also computes the overall dot ball percentage and the variance of dot ball percentage over time.

**Inputs:**  
- `group`: A DataFrame containing bowler statistics, including columns like `start_date`, `balls_bowled`, and `dot_balls_as_bowler`.

**Outputs:**  
A DataFrame with additional columns:
- `longterm_dot_ball_percentage`: Long-term dot ball percentage using an expanding sum.
- `dot_ball_percentage`: Dot ball percentage for each match.
- `longterm_var_dot_ball_percentage`: Variance of the dot ball percentage over time.

#### `calculate_rolling_batting_stats_test()`
**Description:** The `calculate_rolling_batting_stats_test` function calculates rolling batting statistics such as averages, strike rates, and boundary percentages for a player over different rolling window sizes. It applies conditional logic based on a minimum number of balls faced and computes these metrics for three different window sizes: `n1`, `n2`, and `n3`.

**Inputs:**  
- `group`: A DataFrame containing batting statistics, including `runs_scored`, `balls_faced`, `player_out`, `fours_scored`, and `sixes_scored`.  
- `n1` (default=3): Window size for the first rolling window.  
- `n2` (default=7): Window size for the second rolling window.  
- `n3` (default=12): Window size for the third rolling window.  
- `min_balls` (default=20): Minimum number of balls faced to calculate valid strike rates.

**Outputs:**  
A DataFrame with additional columns:
- Batting averages over the windows
- Strike rates over the windows
- Boundary percentages over the windows

#### `calculate_rolling_bowling_stats_test()`
**Description:** The `calculate_rolling_bowling_stats_test` function calculates rolling bowling statistics such as bowling averages, economy rates, strike rates, and an updated CBR for a player over different rolling window sizes. It also computes fielding points based on catches, stumpings, and run outs.

**Inputs:**  
- `group`: A DataFrame containing bowling statistics, including `runs_conceded`, `wickets_taken`, `balls_bowled`, `balls_per_over`, `catches_taken`, `stumpings_done`, `run_out_direct`, and `run_out_throw`.  
- `n1` (default=3): Window size for the first rolling window.  
- `n2` (default=7): Window size for the second rolling window.  
- `n3` (default=12): Window size for the third rolling window.  

**Outputs:**  
A DataFrame with additional columns:
- Bowling averages over the windows
- Economy rates over the windows
- Bowling strike rates over the windows
- A computed bowling CBR value based on the rolling statistics from `n2`.
- Fielding points calculated from rolling aggregates of fielding events (catches, stumpings, and run outs).

#### `calculate_alpha_batsmen_score()`
**Description:**  
The `calculate_alpha_batsmen_score` function calculates the tailored α_batsmen_score for Dream11 point prediction in ODIs over multiple rolling time horizons (n1, n2, n3). It considers factors such as runs scored, strike rate, boundary counts, half-centuries, centuries, and ducks to compute a performance score for each player.

**Inputs:**  
- `group`: A DataFrame containing batting statistics such as `runs_scored`, `strike_rate_n1`, `sixes_scored`, `fours_scored`, `half_centuries_cumsum`, `centuries_cumsum`, and `rolling_ducks_n1`.  
- `n1` (default=3): Window size for the first rolling window (time horizon 1).  
- `n2` (default=7): Window size for the second rolling window (time horizon 2).  
- `n3` (default=12): Window size for the third rolling window (time horizon 3).  

**Outputs:**  
A DataFrame with additional columns:
- Rolling averages of runs scored over the windows.
- Rolling averages of strike rate over the windows
- Rolling averages of sixes scored over the windows
- Rolling averages of fours scored over the windows
- Rolling sums of half-centuries scored over the windows
- Rolling sums of centuries scored over the windows
- Rolling sums of ducks over the windows
- The computed α_batsmen_score for each time window, considering runs, strike rate, boundaries, half-centuries, centuries, and ducks.

#### `calculate_alpha_bowler_score()`
**Description:**  
The `calculate_alpha_bowler_score` function calculates the tailored α_bowler_score for Dream11 point prediction in ODIs over multiple rolling time horizons (n1, n2, n3). It factors in wickets, bowling average, strike rate, economy rate, and maidens to compute a performance score for each bowler.

**Inputs:**  
- `group`: A DataFrame containing bowling statistics.
- `n1`: Window size for the first rolling window (time horizon 1).  
- `n2`: Window size for the second rolling window (time horizon 2).  
- `n3`: Window size for the third rolling window (time horizon 3).  

**Outputs:**  
A DataFrame with additional columns:
- Rolling averages of wickets taken over the windows
- Rolling averages of bowling average over the windows
- Rolling averages of bowling strike rate over the windows
- Rolling averages of economy rate over the windows
- Rolling sums of maidens bowled over the windows
- The computed α_bowler_score for each time horizon, considering wickets, strike rate, economy rate, maidens, and bowling average.

#### `Class FeatureGeneration`
**Description:** Dataframe wrapper class which contains feature generation methods and formulae. Encorporates customisation based on match format.
**Input:**  
- `mw_overall`: Match-wise overview data. 
- `mw_pw_profile`: Match-wise, player-wise data.
- `match_format`: `'Test'`, `'ODI'` or `'T20'`


#### `process_country_and_homeaway()`
**Description:**  
This function processes cricket match data to determine the country of the venue (`country_ground`) and whether a player's team played in a home, away, or neutral match (`home_away`). It utilizes geonames and pycountry libraries for mapping cities to countries.

**Updates:**  
Updates `self.mw_pw_profile` by adding:
- `country_ground`: The country of the ground where the match was played.
- `home_away`: Whether the match was played at home, away, or a neutral venue.


#### `calculate_match_level_venue_stats()`
**Description:**  
Calculates venue-level match statistics (ARPO, boundary percentages, strike rates, and pitch classification) using rolling and cumulative metrics grouped by venue and match type.

**Updates:**  
Updates `self.mw_pw_profile` with:
- `ARPO_venue`: Average Runs Per Over at the venue.
- `Boundary_Percentage_venue`: Percentage of runs from boundaries.
- `BSR_venue`: Batting strike rate at the venue.
- `AFIS_venue`: Average first-innings score at the venue.
- `Pitch_Type`: Classification as Bowling-Friendly, Neutral, or Batting-Friendly.

#### `avg_of_opponent()`
**Description:**  
This function calculates the average fantasy score of opponents faced by a player in cricket matches, based on their role (batter, bowler, all-rounder, or neither). The scores are computed using data grouped by player ID, match type, and opposition team.

**Outputs:**  
Updates `self.mw_pw_profile` with a new column:
- `avg_of_opponent`: The calculated average opponent fantasy score for each player-role and match combination.