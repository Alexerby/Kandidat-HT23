import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
import warnings
from matplotlib.ticker import FuncFormatter


def extract_stibor(path='../Färdig data/STIBOR.xlsx', sheet='Sheet1', col='Tom/Next'):
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

        # Convert values to percentages (divide by 100)
        df[col] = df[col] / 101

        return df[col]

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return pd.DataFrame()
    except pd.errors.SheetNameNotFound:
        print(f"Error: Sheet '{sheet}' not found in the Excel file.")
        return pd.DataFrame()



def save_fig(save_path):
    '''
    Kan användas vid nedladdning av figurer,
    används ej för nuvarande
    '''
    if save_path:
        plt.savefig(save_path)


def is_major_bank(account):
    """
    Funktion för att avgöra om banken är stor eller inte.
    -------------------------------------------------

    param
    ---------
    konto: Skicka en bank som argument.

    return
    ---------
    ret: Returnerar True om banken är stor, annars False.
    """
    
    major_banks = [
        "Handelsbanken: Sparkonto",
        "SEB: Enkla Sparkontot",
        "Swedbank: e-sparkonto",
        "Nordea: Förmånskonto"
    ]
    return 1 if account in major_banks else 0


def get_data(path):
    """
    Funktion för att hämta och förbereda data från en Excel-fil.

    param
    ---------
    sökväg: Sträng som anger sökvägen till Excel-filen.

    return
    ---------
    df: En pandas DataFrame med data från 'Sheet1', där 'Account' och 'Date' är satta som index.
        En ny kolumn 'DummyLargeBank' skapas med hjälp av 'Account'-nivån och funktionen is_major_bank.
    """
    path = path
    df = pd.read_excel(path, sheet_name="Sheet1")

    # Set the 'Account' level as the index
    df = df.set_index(['Account', 'Date'])

    # Skapa 'DummyLargeBank' kolumn efter 'Account'
    df['DummyLargeBank'] = df.index.get_level_values('Account').map(is_major_bank)
    return df


def plot_results(df):
    """
    Funktion för att plotta regressionsresultat baserat på bankstorlek.

    param
    ---------
    df: En pandas DataFrame med relevanta data, inklusive 'DepositRate', 'PolicyRate' och 'DummyLargeBank'.

    return
    ---------
    Ingen retur. Plottar resultaten av regressionsanalysen för stora och små banker.
    """
    plt.figure(figsize=(10, 6))  
    
    # Separera dataframe baserad på 'DummyLargeBank' värde
    df_large_bank = df[df['DummyLargeBank'] == 1]
    df_small_bank = df[df['DummyLargeBank'] == 0]
    
    df['InteractionTerm'] = df['DummyLargeBank'] * df['PolicyRate']
    
    # Definiera modell för storbanker
    model_large_bank = PanelOLS(dependent = df_large_bank['DepositRate'], 
                                exog = df_large_bank[['PolicyRate', 'InteractionTerm']], 
                                entity_effects=True, 
                                check_rank=False,
                                drop_absorbed=True
                                )
    
    # Definiera modell för nisch-/neobanker
    model_small_bank = PanelOLS(dependent = df_small_bank['DepositRate'], 
                                exog = df_small_bank[['PolicyRate', 'InteractionTerm']], 
                                entity_effects=True, 
                                check_rank=False,
                                drop_absorbed=True
                                )
    
    # Passa modell för stor- och nisch-/neobanker
    results_large_bank = model_large_bank.fit(cov_type='clustered', 
                                              cluster_entity=True, 
                                              cluster_time=True, 
                                              use_lsdv=True
                                            )
    
    results_small_bank = model_small_bank.fit(cov_type='clustered', 
                                              cluster_entity=True, 
                                              cluster_time=True, 
                                              use_lsdv=True
                                              )
    # Plotta regression för linjer DummyLargeBank == 1 och DummyLarbeBank == 0
    plt.plot(df_large_bank['PolicyRate'], results_large_bank.predict(), label='Storbanker', color='#4D1355', linestyle='--')
    plt.plot(df_small_bank['PolicyRate'], results_small_bank.predict(), label='Mindre banker', color='#1B5513', linestyle='--')
    
    #  Grafformattering
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.xlabel("Styrränta", fontsize=14)
    plt.ylabel("Inlåningsränta", fontsize=14)
    plt.title(f"Regression", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.legend(fontsize=12)
    
    plt.show()

def plot_average_time_series(file_path):
    """
    Funktion för att plotta genomsnittliga tidsserier baserat på bankstorlek.

    param
    ---------
    file_path: Sträng som anger sökvägen till Excel-filen med relevanta data.
               Förväntar sig data med kolumner som 'Date', 'DepositRate', 'PolicyRate' och 'DummyLargeBank'.

    return
    ---------
    Ingen retur. Plottar resultaten av regressionsanalysen för stora och små banker.
    """

    df_deposit_rates = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Filtered Deposit Rates")

    # Sätt datum kolumn som index 
    df_deposit_rates.set_index('Date', inplace=True)

    # Konvertera kolumner till numerisk och ersätt (ersätt ',' med '.')
    for col in df_deposit_rates.columns:
        if df_deposit_rates[col].dtype == 'O':  
            df_deposit_rates[col] = pd.to_numeric(df_deposit_rates[col].str.replace(',', '.'))

    # Separera dataframes baserad på om banken är klassifierad som storbank eller ej 
    df_major_bank = df_deposit_rates[df_deposit_rates.columns[df_deposit_rates.columns.map(is_major_bank) == 1]]
    df_minor_bank = df_deposit_rates[df_deposit_rates.columns[df_deposit_rates.columns.map(is_major_bank) == 0]]

    df_deposit_rates['AverageLargeBanks'] = df_major_bank.mean(axis=1)
    df_deposit_rates['AverageSmallBanks'] = df_minor_bank.mean(axis=1)

    df_policy_rate = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Policy Rate")

    df_policy_rate.set_index('Date', inplace=True)

    df_vwadr = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="VWADR")

    df_vwadr.set_index('Date', inplace=True)

    # Plot-inställningar
    plt.figure(figsize=(10, 6))
    plt.plot(df_deposit_rates.index, df_deposit_rates['AverageLargeBanks'], label='Medelvärde storbanker', color='#552C13')
    plt.plot(df_deposit_rates.index, df_deposit_rates['AverageSmallBanks'], label='Medelvärde mindre banker', color='#1B5513')
    plt.plot(df_deposit_rates.index, extract_stibor(), label='STIBOR (T/N)', color='red')
    plt.plot(df_policy_rate.index, df_policy_rate['Rate'], label='Styrränta', color='#133C55')
    plt.plot(df_vwadr.index, df_vwadr['Rate'], label='Inlåningsränta banker (SCB)', color='#4D1355')

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentage

    plt.title('Inlåningsräntor, styrränta, STIBOR & genomsnittlig inlåningsränta')
    plt.xlabel('Tid')
    plt.ylabel('Ränta (%)')
    plt.legend()
    plt.show()

    
def plot_shaded_area_with_percentiles_median_and_policy_rate(file_path):
    """
    Funktion för att plotta ett skuggat område med percentiler, median och styrränta.

    param
    ---------
    file_path: Sträng som anger sökvägen till Excel-filen med relevanta data.
               Förväntar sig data med kolumner som 'Date', 'DepositRate' och 'PolicyRate'.

    return
    ---------
    Ingen retur. Plottar resultatet med skuggat område, percentiler, median inlåning och styrränta.
    """
    # Läs in data från kalkylbladet 'Filtered Deposit Rates' till en DataFrame
    df_deposit_rates = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Filtered Deposit Rates")

    # Sätt 'Date'-kolumnen som index
    df_deposit_rates.set_index('Date', inplace=True)

    # Konvertera kolumnerna till numeriska värden (ersätt ',' med '.')
    for col in df_deposit_rates.columns:
        # Kontrollera om kolumnen innehåller strängvärden innan konvertering
        if df_deposit_rates[col].dtype == 'O':  # 'O' representerar Object (sträng) dtype
            df_deposit_rates[col] = pd.to_numeric(df_deposit_rates[col].str.replace(',', '.'))

    # Beräkna percentiler för skuggning
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(df_deposit_rates, percentiles, axis=1)

    # Beräkna medianvärden
    median_values = df_deposit_rates.median(axis=1)

    # Läs in data från kalkylbladet 'Policy Rate' till en separat DataFrame
    df_policy_rate = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Policy Rate")
    df_policy_rate.set_index('Date', inplace=True)

    # Plotta en skuggad area mellan percentilerna
    plt.figure(figsize=(10, 6))
    
    # Skugga området mellan 10:e och 90:e percentilen med en ljusare skugga
    plt.fill_between(df_deposit_rates.index, 
                     percentile_values[0], 
                     percentile_values[-1], 
                     color='gray', alpha=0.2, 
                     label='Inlåningsränta (10:e-90:e percentilen)')

    # Skugga området mellan 25:e och 75:e percentilen med en mörkare skugga
    plt.fill_between(df_deposit_rates.index, 
                     percentile_values[1], 
                     percentile_values[3], 
                     color='gray', 
                     alpha=0.4, label='Inlåningsränta (25:e-75:e percentilen)')

    # Plotta medianvärdet som en tidsserielinje
    plt.plot(df_deposit_rates.index, median_values, color='#133455', label='Median inlåningsränta')

    # Plotta styrräntan som en tidsserielinje
    plt.plot(df_policy_rate.index, df_policy_rate['Rate'], color='#218c74', label='Styrränta', linestyle='--')

    plt.title('Skuggad area med inlåningsräntor med percentiler, median och styrränta')
    plt.xlabel('Tid')
    plt.ylabel('Ränta (%)')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentage
    plt.show()

def interaction_plot(df, interaction_col, x_col, y_col):
    """
    Creates an interaction plot to visualize the relationship between two variables across levels of a third variable.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        interaction_col (str): Name of the column representing the interaction term.
        x_col (str): Name of the column representing the variable on the x-axis.
        y_col (str): Name of the column representing the dependent variable on the y-axis.
    """

    plt.figure(figsize=(10, 6))

    sns.relplot(x=x_col, y=y_col, hue=interaction_col, data=df, palette='viridis', kind='line')

    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.title(f"Interaction Plot: {x_col} vs. {y_col} across {interaction_col}", fontsize=14)
    plt.show()
    


def main():
    df = get_data("../Färdig Data/Melted df.xlsx")
    df_file_path = "../Färdig Data/Färdig data.xlsx"

    # Definiera interaktionsterm
    df['InteractionTerm'] = df['DummyLargeBank'] * df['PolicyRate']

    # Definiera modell
    model = PanelOLS(dependent=df['DepositRate'], 
                     exog=df[['PolicyRate', 'InteractionTerm']], 
                     entity_effects=True,
                     drop_absorbed=True,
                     check_rank=False,
                    )  

    # Passa modell
    results = model.fit(cov_type='clustered', 
                        cluster_entity=True, 
                        cluster_time=True,
                        use_lsdv=True
                        )

    # Plottnings av grafer 
    plot_shaded_area_with_percentiles_median_and_policy_rate(df_file_path)
    plot_results(df)  # Plot the scatter plot of Policy Rate vs. Deposit Rate
    plot_average_time_series(df_file_path)

    # Printa resultat av regression
    print(results)

# Kör scriptet ifall den körs direkt som modul (inte som importerat paket)
if __name__ == "__main__":
    main()

