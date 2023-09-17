from requests import get as req_get
import bs4
import re
import sys
from zipfile import ZipFile
import os
import pandas as pd
import atexit
import exit_handling
from shutil import rmtree


def exiting():
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        print('WARNING: last download my be corrupted if program was interrupted during download or extraction',
              file=sys.stderr)

    if not os.path.exists('temp_data'):
        return
    print('Cleaning up temporary files...')
    rmtree('temp_data')


def format_csv(file_path: str, output_path: str, start_date: str) -> bool:
    df: pd.DataFrame = pd.read_csv(file_path,
                                   skiprows=4,  # skip metadata of csv
                                   sep=';',  # separator
                                   skipinitialspace=True,  # remove trailing whitespace
                                   na_values=['EOR', -999],  # End Of Record is irrelevant, -999 means missing value
                                   low_memory=False,  # warning about mixed types
                                   )
    df.columns = df.columns.str.strip()  # remove trailing whitespaces
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H%M')  # convert to datetime
    df.index = df['Time']  # set index to datetime
    if start_date not in df.index:
        return False
    df = df[start_date:]  # using data from 2015 onwards
    df.drop('Time', axis=1, inplace=True)  # remove unnecessary column
    df.dropna(how='all', axis=1, inplace=True)  # remove columns with all NaN values
    df.drop(['StationNumber', 't', 'tn', 'tx', 'v', 'p', 'fs', 'fsd', 'fx', 'fxd', 'fxdat', 'fd', 'et5', 'et10', 'et20',
             'et50', 'et100', 'tsn', 'suv'], axis=1, inplace=True, errors='ignore')
    # 'suv' column doesn't exist in some instances
    # still deciding if I should keep the 'we' column
    df.to_csv(output_path, sep=';')
    return True


def main():
    # Create directories
    if not os.path.exists('omsz_data'):
        os.makedirs('omsz_data')
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')

    # Metadata download
    omsz_meta_url = 'https://odp.met.hu/climate/observations_hungary/hourly/station_meta_auto.csv'
    omsz_meta_page = req_get(omsz_meta_url)
    omsz_meta_path = 'temp_data/station_meta_auto.csv'
    if omsz_meta_page.status_code == 200:
        with open(omsz_meta_path, 'wb') as f:
            f.write(omsz_meta_page.content)
            print(f"Downloaded metadata: {omsz_meta_path}")
    else:
        print(f"Error {omsz_meta_page.status_code} for {omsz_meta_url}", file=sys.stderr)
        print("Can't continue, exiting...", file=sys.stderr)
        sys.exit(1)

    # Get historical data download links
    omsz_historical_url = 'https://odp.met.hu/climate/observations_hungary/hourly/historical/'
    omsz_historical_page = req_get(omsz_historical_url)
    omsz_historical_soup = bs4.BeautifulSoup(omsz_historical_page.text, 'html.parser')

    file_download = omsz_historical_soup.find_all('a')
    regex = re.compile(r'.*\.zip')
    file_download = [link.get('href').strip() for link in file_download if regex.match(link.get('href'))]
    file_download = list(set(file_download))
    regex = re.compile(r'.*20221231.*')
    file_download = [f"{omsz_historical_url}{file}" for file in file_download if regex.match(file)]

    # Load metadata
    meta = pd.read_csv(omsz_meta_path, sep=';', skipinitialspace=True, na_values='EOR')
    meta.columns = meta.columns.str.strip()
    meta.index = meta['StationNumber']
    meta.drop('StationNumber', axis=1, inplace=True)
    meta.dropna(how='all', axis=1, inplace=True)
    meta = meta[~meta.index.duplicated(keep='last')]
    meta['StartDate'] = pd.to_datetime(meta['StartDate'], format='%Y%m%d')
    meta['EndDate'] = pd.to_datetime(meta['EndDate'], format='%Y%m%d')
    # I'll save metadata to csv, so I can use it later
    meta.to_csv('omsz_meta.csv', sep=';')
    os.remove(omsz_meta_path)

    start_date: str = '2015-01-01 00:00:00'
    # Download and extract historical data
    for link in file_download[:2]:
        file = link.split('/')[-1]
        temp_path = f"temp_data/{file}"
        response = req_get(link, timeout=60)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded {temp_path}")
        else:
            print(f"Error {response.status_code} for {link}", file=sys.stderr)
            continue

        regex = re.compile(r'.*_(\d{5})_.*')
        try:
            with ZipFile(temp_path, 'r') as zip_obj:
                # extract all files to 'omsz_data' and remove redundant directory
                unzipped_path = f"temp_data/{file.split('.')[0]}"
                zip_obj.extractall(unzipped_path)

                csv_code = int(regex.match(unzipped_path).group(1))
                csv_name = f"{meta['RegioName'][csv_code].strip()}_{meta['StationName'][csv_code].strip()}.csv "
                csv_path = f"{unzipped_path}/{os.listdir(unzipped_path)[0]}"
                if format_csv(csv_path, f"omsz_data/{csv_name}", start_date):
                    print(f"Extracted and formatted: {csv_name}")
                else:
                    print(f"Throwing away: {csv_name},"
                          f"\tREASON: station started recording data later than {start_date}",
                          file=sys.stderr, sep='\n')
        # need to detect all possible exceptions with ZipFile and os functions
        except Exception as e:
            print(f"Error {e} for {temp_path}", file=sys.stderr)
            continue


if __name__ == '__main__':
    atexit.register(exiting)
    main()
