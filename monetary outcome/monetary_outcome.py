import os
import pandas as pd

def interpolering(column: str, sheet: str, path: str, frequency: str = 'D'):
    '''
    Interpolerar en given dataframe till daglig data

    Parametrar:
    - column (str): namn på kolumn
    - sheet (str): namn på excelbladet
    - path (str): namn på sökväg till excelbok
    - frequency (str): Den frekvens som interpoleringen ska ske till

    Returns:
    - dataframe (pandas.DataFrame): interpolerad dataframe
    '''
    
    # Läs in excelfil
    df = pd.read_excel(path, sheet_name=sheet)

    # Gör 'Date' kolumnen till datumformat
    df['Date'] = pd.to_datetime(df['Date'])

    # Indexera efter datum
    df.set_index('Date', inplace=True)

    # Generera datum-spann för dagliga datum
    daily_index = pd.date_range(df.index.min(), df.index.max(), freq=frequency)
    
    df = df.reindex(daily_index)
   
    # Utför linjär interpolering
    df[column] = df[column].interpolate(method='linear', limit_direction='both')

    return df


def policy_rate(path, sheet, column, start_date='2022-04-01', end_date='2023-08-01'):
    df = pd.read_excel(path, sheet_name=sheet)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Filtrera dataframe mellan angivna datum
    df = df.loc[start_date:end_date]

    return df


def outcome(storbank: bool = True, print_outcome: bool = False):
    '''
    Beräknar den aggregerade differensen mellan storbanker och nischbanker
    baserat på koefficienterna från huvudregression.

    Parametrar:
    - storbank (bool):      Om True, använd en större bankmultiplikator med ett interaktionsterm;
                            om False, använd en standardmultiplikator för mindre banker.
    
    - print_outcome (bool): Om True, skriv ut det beräknade utfallet; standard är False.

    Returns:
    - total_outcome (float): Den aggregerade summan av de beräknade utfallen.
    '''

    beta_pr = 0.5377
    beta_interactionterm = 0.1579

    df_akp = interpolering(path='../Färdig data/Avistakontopengar.xlsx',
                            sheet='Avistakontopengar',
                            column='Avistakontopengar')

    df_pr = policy_rate(path='../Färdig data/Färdig data.xlsx',
                        sheet='Policy Rate',
                        column='Rate',
                        start_date='2022-04-01',
                        end_date='2023-08-01')

    # Beräkna multiplikatorn för varje datum
    if storbank:
        multiplier = (1 + df_pr['Rate'] * (beta_pr - beta_interactionterm)) ** (1/365) -1
    else:
        multiplier = (1 + df_pr['Rate'] * beta_pr) ** (1/365) - 1

    # Avistakontopengar \times multiplier
    df_result = df_akp['Avistakontopengar'] * multiplier

    # Skapa ny dataframe för resultat
    result_df = pd.DataFrame({'Outcome': df_result})

    # Aggregerad summa
    total_outcome = result_df['Outcome'].sum()

    return total_outcome


def main():
    print(f'\nStorbank: {outcome(storbank=True, print_outcome=True)} MSEK')
    print(f'Nischbank: {outcome(storbank=False, print_outcome=True)} MSEK')

    diff = outcome(storbank=False) - outcome(storbank=True)
    print(f'Skillnad = (Nischbank - Storbank) = {diff} MSEK')

# Kör om kod körs direkt som skript och inte importerat bibliotek
if __name__ == '__main__':
    main()

