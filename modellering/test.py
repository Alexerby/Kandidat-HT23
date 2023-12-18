import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
import warnings
from matplotlib.ticker import FuncFormatter


def extract_stibor(path, sheet, col: str = 'Tom/Next'):
    try:
        df = pd.read_excel(path, 
                           sheet_name=sheet,
                           header=2,
                           index_col='Date',
                           usecols=['Date', col])

        # Reindex to include the desired start date
        start_date = '2017-12-29'
        end_date = '2023-08-01'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = df.reindex(date_range)

        # Perform daily interpolation
        df = df.interpolate(method='linear', axis=0)

        # Filter the DataFrame for the specified date range
        start_date = '2018-01-01'
        df = df.loc[start_date:end_date]

        print(df)

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except pd.errors.SheetNameNotFound:
        print(f"Error: Sheet '{sheet}' not found in the Excel file.")



extract_stibor(
        path = '../Färdig data/STIBOR.xlsx',
        sheet = 'Monthly statistics'
        )
