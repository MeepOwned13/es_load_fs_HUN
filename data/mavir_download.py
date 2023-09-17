from requests import get as req_get
import sys
import os
import pandas as pd
import atexit
import exit_handling
from datetime import datetime


def exiting():
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        print("WARNING: download may be corrupted, because program was interrupted or an exception occured!",
              file=sys.stderr)
    else:
        if not os.path.exists('temp_data'):
            return
        print('Cleaning up temporary files...')
        for file in os.listdir('temp_data'):
            os.remove(f"temp_data/{file}")
        os.rmdir('temp_data')


def main():
    from_time = pd.to_datetime('2022-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    from_time = from_time - pd.Timedelta(hours=2)  # we need to subtract 2 hours from the date because of the timezone
    from_in_ms = int(from_time.value / 1e6)

    to_time = datetime.now()
    to_time = to_time.replace(minute=0, second=0, microsecond=0)  # request goes till end of current day
    to_time = pd.to_datetime(to_time, format='%Y-%m-%d %H:%M:%S') - pd.Timedelta(hours=2)
    to_in_ms = int(to_time.value / 1e6)

    url = (f"https://www.mavir.hu/rtdwweb/webuser/chart/7678/export"
           f"?exportType=xlsx"
           f"&fromTime={from_in_ms}"
           f"&toTime={to_in_ms}"
           f"&periodType=hour"
           f"&period=1")

    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')

    temp_path = f"temp_data/mavir.xlsx"
    response = req_get(url, timeout=120)
    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            print(f"Downloaded {temp_path}")
    else:
        print(f"Error {response.status_code} for request", file=sys.stderr)
        return 1

    df = pd.read_excel(temp_path, skiprows=0, parse_dates=True, engine='openpyxl')
    df.to_csv('mavir_data.csv', index=False, sep=';')


# only run if executed
if __name__ == '__main__':
    atexit.register(exiting)
    main()
