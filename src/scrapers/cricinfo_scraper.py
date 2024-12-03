import os
import argparse
import requests
from bs4 import BeautifulSoup


def scrape_cricinfo_player(cricinfo_id: str):
    """
    Scrape player details from ESPN Cricinfo using the player's unique ID.
    """

    url = f"https://www.espncricinfo.com/cricketers/player-{cricinfo_id}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        #write soup to /Users/sunrajpatel/Desktop/code/D11_Sample_Structure/src/data/interim/test.txt
        with open('/Users/sunrajpatel/Desktop/code/D11_Sample_Structure/src/data/interim/test.txt', 'w') as f:
            f.write(soup.prettify())
        
        
        # Initialize variables
        full_name = batting_style = bowling_style = playing_role = None
        
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
        return cricinfo_id, full_name, batting_style, bowling_style, playing_role
    else:
        return cricinfo_id, None, None, None, None
    

# Test the function with a sample player ID, 253802
if __name__ == "__main__":
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    output_path = this_file_dir + 'mock_outputs/cricinfo_player_scrape.txt'

    parser = argparse.ArgumentParser(description="Scrape player profile for a Player profile.")
    parser.add_argument('--id', type=str, help="Player id to be scraped", default="253802")
    args = parser.parse_args()

    player_id = args.id
    player_data = scrape_cricinfo_player(player_id)

    with open(output_path, "w") as f:
        f.write(str(player_data))
    print(f"Scraped player data for player ID {player_id}")
