import os
import requests

def execute_scraper():
    json_url = "https://cricsheet.org/downloads/all_json.zip"
    csv_url = "https://cricsheet.org/downloads/all_csv2.zip"
    people_csv_url = "https://cricsheet.org/register/people.csv"


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



if __name__ == "__main__":
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

    target_json_dir = this_file_dir + 'mock_outputs/cricksheet/json/'
    target_csv_dir = this_file_dir + 'mock_outputs/cricksheet/csv/'
    target_people_csv_path = this_file_dir + "mock_outputs/cricksheet/people.csv"

    execute_scraper()

