from requests import get as req_get
import sys
import os
import pandas as pd
import atexit
import exit_handling
from datetime import datetime
import warnings
from shutil import rmtree


def exiting():
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        print("WARNING: download may be corrupted, because program was interrupted or an exception occured!",
              file=sys.stderr)

    if not os.path.exists('temp_data'):
        return
    print('Cleaning up temporary files...')
    rmtree('temp_data')


def format_mavir(dataframe: pd.DataFrame):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe['Time'] = (pd.to_datetime(dataframe['Időpont'], utc=True)).dt.tz_localize(None)
    dataframe['Time'] = dataframe['Time']
    dataframe.index = dataframe['Time']
    dataframe.drop(['Time', 'Időpont'], axis=1, inplace=True)
    dataframe.dropna(axis=0, inplace=True)
    dataframe.drop(['Nettó terv rendszerterhelés', 'Bruttó hitelesített rendszerterhelés tény',
                    'Nettó tény rendszerterhelés - net.ker.elsz.meres', 'Bruttó terv rendszerterhelés',
                    'Bruttó tény rendszerterhelés', 'Nettó rendszerterhelés tény - üzemirányítási',
                    'Nettó terv rendszertermelés', 'Nettó MAVIR rendszerterhelés becslés'],
                    inplace=True, axis=1)
    return dataframe


def download_from_to(name: str, from_time: int, to_time: int):
    """Time should be given as ms"""
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    if not os.path.exists('mavir_data'):
        os.makedirs('mavir_data')

    url = (f"https://www.mavir.hu/rtdwweb/webuser/chart/7678/export"
           f"?exportType=xlsx"
           f"&fromTime={from_time}"
           f"&toTime={to_time}"
           f"&periodType=hour"
           f"&period=1")

    temp_path = f"temp_data/{name}.xlsx"
    response = req_get(url, timeout=120)
    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            print(f"Downloaded {temp_path}")
    else:
        print(f"Error {response.status_code} for request", file=sys.stderr)
        return 1

    # suppress default style warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(temp_path, skiprows=0, parse_dates=True, engine='openpyxl')

    return format_mavir(df)


def main():
    from_time = pd.to_datetime('2015-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    from_time = from_time - pd.Timedelta(hours=2)  # we need to subtract 2 hours from the date because of the timezone
    from_in_ms = int(from_time.value / 1e6)

    to_time = datetime.now()
    to_time = to_time.replace(minute=0, second=0, microsecond=0)  # request goes till end of current day
    to_time = pd.to_datetime(to_time, format='%Y-%m-%d %H:%M:%S') - pd.Timedelta(hours=2)
    to_in_ms = int(to_time.value / 1e6)

    # i'll split the request into 2 parts, because the request fails if it's too long (more than 60_000 lines)
    # realistically, we will never need more than 120_000 lines at once
    middle_time = from_time + (to_time - from_time) / 2
    middle_time = pd.to_datetime(middle_time)
    middle_time = middle_time.replace(minute=0, second=0, microsecond=0)
    middle_in_ms = int(middle_time.value / 1e6)

    df1 = download_from_to('mavir_1', from_in_ms, middle_in_ms)
    df2 = download_from_to('mavir_2', middle_in_ms, to_in_ms)

    pd.concat([df1, df2]).to_csv('mavir_data/mavir.csv', sep=';')

    if not os.path.exists('mavir_data'):
        os.makedirs('mavir_data')


# only run if executed
if __name__ == '__main__':
    atexit.register(exiting)
    main()
