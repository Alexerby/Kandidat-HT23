import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker


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

    # Create the "DummyLargeBank" column using the 'Account' level
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
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    
    # Separate the DataFrame based on 'DummyLargeBank' values
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
    
    # Plot the regression lines for DummyLargeBank == 1 and DummyLargeBank == 0
    plt.plot(df_large_bank['PolicyRate'], results_large_bank.predict(), label='Storbanker', color='#4D1355', linestyle='--')
    plt.plot(df_small_bank['PolicyRate'], results_small_bank.predict(), label='Mindre banker', color='#1B5513', linestyle='--')
    
    # Formatting for the graph
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.xlabel("Styrränta", fontsize=14)
    plt.ylabel("Inlåningsränta", fontsize=14)
    plt.title(f"Regression", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.2)  # Add grid lines
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
    # Read the 'Filtered Deposit Rates' data into a DataFrame
    df_deposit_rates = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Filtered Deposit Rates")

    # Set the 'Date' column as the index
    df_deposit_rates.set_index('Date', inplace=True)

    # Convert columns to numeric (replace ',' with '.')
    for col in df_deposit_rates.columns:
        # Check if the column contains string values before conversion
        if df_deposit_rates[col].dtype == 'O':  # 'O' represents Object (string) dtype
            df_deposit_rates[col] = pd.to_numeric(df_deposit_rates[col].str.replace(',', '.'))

    # Separate the DataFrame based on the 'is_major_bank' function
    df_major_bank = df_deposit_rates[df_deposit_rates.columns[df_deposit_rates.columns.map(is_major_bank) == 1]]
    df_minor_bank = df_deposit_rates[df_deposit_rates.columns[df_deposit_rates.columns.map(is_major_bank) == 0]]

    # Create two new columns for average values
    df_deposit_rates['AverageLargeBanks'] = df_major_bank.mean(axis=1)
    df_deposit_rates['AverageSmallBanks'] = df_minor_bank.mean(axis=1)

    # Read the 'Policy Rate' data into a separate DataFrame
    df_policy_rate = pd.read_excel(file_path, parse_dates=['Date'], sheet_name="Policy Rate")

    # Set the 'Date' column as the index
    df_policy_rate.set_index('Date', inplace=True)

    # Plot the average time series and policy rate
    plt.figure(figsize=(10, 6))
    plt.plot(df_deposit_rates.index, df_deposit_rates['AverageLargeBanks'], label='Medelvärde storbanker', color='#552c13')
    plt.plot(df_deposit_rates.index, df_deposit_rates['AverageSmallBanks'], label='Medelvärde mindre banker', color='#133455')
    plt.plot(df_policy_rate.index, df_policy_rate['Rate'], label='Styrränta', color='#218c74')

    plt.title('Medelvärde inlåningsräntor (och styrränta)')
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
    plt.fill_between(df_deposit_rates.index, percentile_values[0], percentile_values[-1], color='gray', alpha=0.2, label='Inlåningsränta (10:e-90:e percentilen)')

    # Skugga området mellan 25:e och 75:e percentilen med en mörkare skugga
    plt.fill_between(df_deposit_rates.index, percentile_values[1], percentile_values[3], color='gray', alpha=0.4, label='Inlåningsränta (25:e-75:e percentilen)')

    # Plotta medianvärdet som en tidsserielinje
    plt.plot(df_deposit_rates.index, median_values, color='#133455', label='Median inlåning')

    # Plotta styrräntan som en tidsserielinje
    plt.plot(df_policy_rate.index, df_policy_rate['Rate'], color='#218c74', label='Styrränta', linestyle='--')

    plt.title('Skuggad area med inlåningsräntor med percentiler, median och styrränta')
    plt.xlabel('Tid')
    plt.ylabel('Ränta (%)')
    plt.legend()
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

    # Use seaborn's `relplot` for an interaction plot
    sns.relplot(x=x_col, y=y_col, hue=interaction_col, data=df, palette='viridis', kind='line')

    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.title(f"Interaction Plot: {x_col} vs. {y_col} across {interaction_col}", fontsize=14)

    plt.show()
    


def main():
    # Ladda in din data
    df = get_data("../Färdig Data/Melted df.xlsx")
    df_file_path = "../Färdig Data/Färdig data.xlsx"

    # Plotta skuggat område med percentiler, median och styrränta
    plot_shaded_area_with_percentiles_median_and_policy_rate(df_file_path)

    # Skapa en interaktionsterm mellan "DummyLargeBank" och "PolicyRate"
    df['InteractionTerm'] = df['DummyLargeBank'] * df['PolicyRate']

    # Utför fixed-effects regression med entity fixed effects 
    model = PanelOLS(dependent=df['DepositRate'], 
                     exog=df[['PolicyRate', 'InteractionTerm']], 
                     entity_effects=True,
                     drop_absorbed=True,
                     check_rank=False,
                    )  

    # Specificera modell med klustrad entity och tid
    results = model.fit(cov_type='clustered', 
                        cluster_entity=True, 
                        cluster_time=True,
                        use_lsdv=True
                        )
    
    # Plotta resultaten från regressionen
    plot_results(df)  # Plotta spridningsdiagrammet av Policy Rate vs. Deposit Rate

    # Plotta genomsnittliga tidsserier
    plot_average_time_series(df_file_path)


    # Skriv ut resultaten från regressionen
    print(results)


# Kör scriptet ifall den körs direkt som modul (inte som importerat paket)
if __name__ == "__main__":
    main()

