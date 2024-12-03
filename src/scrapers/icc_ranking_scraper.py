import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter("ignore", InsecureRequestWarning)


# Scraper function
def scrape_rankings(START_DATE, END_DATE):
    MATCH_TYPES = ["test", "odi", "t20"]
    RANKING_TYPES = ["batting", "bowling"]
    BASE_URL = "https://www.relianceiccrankings.com/datespecific/{match_type}/{ranking_type}/{year}/{month}/{day}/"
    all_data = []
    current_date = START_DATE

    # Use a session for efficiency
    with requests.Session() as session:
        while current_date <= END_DATE:
            date_str_parts = {
                "year": current_date.strftime("%Y"),
                "month": current_date.strftime("%m"),
                "day": current_date.strftime("%d"),
            }
            for match_type in MATCH_TYPES:
                for ranking_type in RANKING_TYPES:
                    url = BASE_URL.format(match_type=match_type, ranking_type=ranking_type, **date_str_parts)
                    print(f"Scraping URL: {url}")

                    try:
                        response = session.get(url, verify=False)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, "html.parser")

                        # Extract ranking table
                        table = soup.find("table")
                        if not table:
                            print(f"No data found for {current_date.date()} - {match_type} - {ranking_type}")
                            continue

                        rows = table.find_all("tr")[1:]  # Skip header row
                        for row in rows:
                            cols = row.find_all("td")
                            if len(cols) < 5:  # Ensure there are enough columns
                                continue
                            player_data = {
                                "Date": current_date.strftime("%Y-%m-%d"),
                                "Match Type": match_type.upper(),
                                "Ranking Type": ranking_type.upper(),
                                "Rank": cols[0].text.strip(),
                                "Player": cols[1].text.strip(),
                                "Country": cols[2].text.strip(),
                                "Rating": cols[3].text.strip(),
                                "Career Best": cols[4].text.strip(),
                            }
                            all_data.append(player_data)

                    except Exception as e:
                        print(f"Error scraping {url}: {e}")

            current_date += timedelta(days=30)  # Increment by 30 days for the next scrape

    # Save data to CSV
    df = pd.DataFrame(all_data)
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(this_file_dir, "./mock_outputs/icc_ranking_scrape.csv")
    # 
    df.to_csv(output_path, index=False)

    print("Scraping completed!")
    return df



# mock call for this function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Scrape rankings for a date range.")
    parser.add_argument('--start', type=str, help="Start date in YYYY-MM-DD format", default="2024-01-01")
    parser.add_argument('--end', type=str, help="End date in YYYY-MM-DD format", default="2024-12-20")
    args = parser.parse_args()

    # Convert string dates to datetime objects
    START_DATE = datetime.strptime(args.start, "%Y-%m-%d")
    END_DATE = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"Scraping ICC ranks from {START_DATE} to {END_DATE}")

    # Call the scraping function with the dates
    scrape_rankings(START_DATE, END_DATE)