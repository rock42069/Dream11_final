# data_download.py


## Functions
### `execute_scraper()`
**Description:** The `execute_scraper` function automates the process of downloading, extracting, and organizing cricket match data from Cricsheet into a structured directory. It handles both the available JSON and CSV data formats, as well as downloading the separate CSV file containing player registry.  

The scraper can be tested in the `/scraper/cricsheet.py` file

**Input:** This function takes no arguments.  
**Output:** This function creates the following structure and files in the local filesystem:
- A directory containing unzipped JSON files from Cricsheet.  
- A directory containing unzipped CSV files from Cricsheet.  
- A CSV file for player metadata

### `import_data(file_path)`
**Description:** Reads a cricsheet JSON file and extracts match info and innings data.  
**Inputs:** 
- `file_path`: Path to the JSON file.  

**Outputs:**
- info_data: General information about the match, extracted from the `info` level in the match json.
- innings_data: List of innings details, extracted from the `innings` level in the match json.

### `fix_dates(dates)`
**Description:** Formats match dates into a string. Handles single and multi-day matches. Made to handle the mixed date formats in cricsheet data.  
**Inputs:**  
- `dates`: List of dates.  

**Output:** Formatted date string.

### `parse_info_data(info_data)`
**Description:** Extracts and formats match-level attributes from info_data.  
**Inputs:** 
- `info_data`: Metadata about the match.  

**Outputs:** Dictionary of parsed match-overview attributes.

### `get_players_data_dict(players_in_match, player_ids, info_dict)`
**Purpose:** Initializes a dictionary for storing player stats with default values.  
**Inputs:**
- `players_in_match`: List of player names in the match.  
- `player_ids`: Dictionary mapping player names to IDs.  
- `info_dict`: Match info dictionary with team-player mappings.  

**Output:** total_data: Dictionary initialized for all players with default stats.  


### `get_overs(session)`
**Description:** Safely extracts overs data from an innings session.  
**Input:**
- `session`: Innings session dictionary.

**Output:** List of overs.

### `is_wicket(ball)`
**Description:** Checks if a ball resulted in a wicket.  
**Input:**
- `ball`: Ball data dictionary.

**Output:** `True` if the ball caused a wicket; otherwise, `False`.

### `get_wicket_data(wicket)`
**Purpose:** Extracts details of a wicket event.  
**Input:**  
- `wicket`: Wicket data dictionary.  

**Output:** Dictionary with details of the dismissed player, dismissal kind, and fielders.

### `split_data(total_data)`
**Purpose:** Splits player stats into batting and bowling datasets.  
**Input:**  
- `total_data`: Dictionary containing player stats.  

**Outputs:**  
- `batsmen_data`: Stats relevant to batting.  
- `bowlers_data`: Stats relevant to bowling.

### `export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order)`
**Purpose:** Appends match data for all players to the CSV file.  
**Inputs:**  
- `batsmen`: Batting stats dictionary.  
- `bowlers`: Bowling stats dictionary.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `batsman_order`: Batting order list.  

**Output:** Saves the current dataframe/JSON to the appropriate output file path.

### `parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, match_info)`
**Purpose:** Processes innings data to compute player stats and then exports to CSV.  
**Inputs:**  
- `innings_data`: List of innings details.  
- `players_in_match`: List of player names in the match.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `player_ids`: Player ID mappings.  
- `match_info`: Match metadata.  

**Output:** None

### `get_players(info_data)`
**Purpose:** Extracts a list of players involved in the match.  
**Input:**  
- `info_data`: Match metadata.  

**Output:** List of player names.

### `generate(file_path)`
**Purpose:** Processes a single JSON file to generate match data in CSV format.  
**Input:**  
- `file_path`: Path to the JSON file.  

**Output:** None

### `json_generator()`
**Purpose:** Iterates through all JSON files in the directory and processes them.  
**Output:** Returns 1 on successful completion.
