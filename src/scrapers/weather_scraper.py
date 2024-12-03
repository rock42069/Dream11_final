import os
import argparse
import optparse
import csv
import numpy as np
import openmeteo_requests
import pandas as pd
from retry_requests import retry

def get_weather_data(latitude, longitude, date, variables=["temperature_2m", "relative_humidity_2m", "dew_point_2m"]):
    """
    Fetch hourly weather data for a specified location and date range.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        date (str): Date in 'YYYY-MM-DD' format.
        variables (list of str): List of requested weather variables (e.g., ["temperature_2m", "relative_humidity_2m"]).

    Returns:
        dict: Dictionary containing the requested weather variables for the specified date and time.
    """
    # Set up the request URL and parameters for Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": variables
    }
    
    # Make the API request
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]  # Process the first location (extendable for multiple locations)

        # Process the hourly data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.to_datetime(hourly.Time(), unit="s", utc=True)
        }
        
        # Extract and store the requested weather variables
        weather_data = {}
        for i, variable in enumerate(variables):
            weather_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        # Add weather data to the hourly_data dictionary
        hourly_data.update(weather_data)

        # Check the length of the data to ensure the index is valid
        data_length = len(hourly_data["temperature_2m"])
        
        # Return data at 3pm (index 15) if available, otherwise return available data
        if data_length > 15:
            temperature = float(hourly_data["temperature_2m"][15])  # 3pm data
            relative_humidity = float(hourly_data["relative_humidity_2m"][15])
            dew_point = float(hourly_data["dew_point_2m"][15])
        else:
            print("Data for 3pm (index 15) is not available. Returning last available data.")
            temperature = float(hourly_data["temperature_2m"][-1])  # Last available data
            relative_humidity = float(hourly_data["relative_humidity_2m"][-1])
            dew_point = float(hourly_data["dew_point_2m"][-1])

        return temperature, relative_humidity, dew_point

    except Exception as e:
        print(f"Error fetching data from Open-Meteo API: {e}")
        return None, None, None


if __name__ == "__main__":
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    output_path = this_file_dir + 'mock_outputs/weather_scrape.txt'

    parser = argparse.ArgumentParser(description="Scrape player profile for a Player profile.")
    parser.add_argument('--lat', type=str, help="Latitude of location to be scraped", default="51.5074")
    parser.add_argument('--lon', type=str, help="Longitude of location to be scraped", default="0.1278")
    parser.add_argument('--date', type=str, help="Date in 'YYYY-MM-DD' format", default="2022-01-01")
    args = parser.parse_args()

    openmeteo = openmeteo_requests.Client()
    latitude = float(args.lat)
    longitude = float(args.lon)
    date = args.date
    weather_data = get_weather_data(latitude, longitude, date)

    with open(output_path, "w") as f:
        f.write(str(weather_data))
    print(f"Scraped weather data for location ({latitude}, {longitude}) on {date}")