from requests import get as req_get
import sys
import os
import pandas as pd
import atexit
import exit_handling


def exiting():
    if hooks.exit_code != 0 or hooks.exception is not None:
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
    pass


# only run if executed
if __name__ == '__main__':
    hooks = exit_handling.ExitHooks()
    hooks.hook()
    atexit.register(exiting)
    main()
