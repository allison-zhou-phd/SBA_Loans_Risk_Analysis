import numpy as np
import pandas as pd
from datetime import datetime 

def pd_columnToNumber(df,col_name):
    for c in col_name:
        df[c] = [stringToNumber(x) for x in df[c]]
    return df

def stringToNumber(s):
    """
    Convert number in accounting format from string to float

    Args:
    s: number as string in accounting format

    Returns:
    float number
    """
    if type(s).__name__=="str":
        s = s.strip()
        if s =="-":
            s = 0
        else:
            s = s.replace(",","").replace("$","")
            if s.find("(")>=0 and s.find(")")>=0:
                s = s.replace("(","-").replace(")","")
    return float(s)

def fix_Date_Year(df, col_1, col_2):
    """Some approval dates are imported as incorrect year (e.g. 2069 instead of 1969).  This function fixes it.
    Args:
        df:     the dataframe 
        col_1:  the col that contains the incorrect date
        col_2:  the col that contains the correct reference year
    Returns:
        df:     updated dataframe
    """
    for idx, date in enumerate(df[col_1]):
        year1 = date.year
        #year2 = df[col_2][idx]
        year2 = df.loc[idx, col_2]
        if np.abs(year1-year2)>95:
            year1 -=100
            #df[idx,col_1]=df[col_1][idx].replace(year=year1)
            df.loc[idx, col_1]=df.loc[idx, col_1].replace(year=year1)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/SBAnational.csv', parse_dates=['ApprovalDate','ApprovalFY'])
    df['Sector'] = [int(str(x)[:2]) for x in df['NAICS']]
    df.loc[df['ApprovalFY']=='1976A','ApprovalFY']='1976'
    df['ApprovalFY'] = df['ApprovalFY'].astype(int)
    df['NewBiz'] = df['NewExist'].map({0:1, 1:0, 2:1})
    df['Franchise'] = [0 if (x==1 or x==0) else 1 for x in df['FranchiseCode']]
    df['RevLine'] = df['RevLineCr'].map({0:0, 1: 1, 2:1, 3:1, 4:1, 5:1, 7:1, 'A':1, 'C':0, 'N':0, 'Q':1, 'R':1, 'T':1, 'Y':1})
    df['LowDocu'] = df['LowDoc'].map({0:0, 1:1, 'N':0, 'Y':1})
    df['Default'] = df['MIS_Status'].map({'P I F':0, 'CHGOFF': 1})
    cols = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
    df = pd_columnToNumber(df, cols)
    df = df.rename(columns={'LoanNr_ChkDgt':'LoanNr', 'NoEmp':'NumEmp', 'DisbursementDate':'DisburseDate', 'DisbursementGross':'DisburseGross'})
    df_loan = df[['LoanNr','State','Bank','BankState','Sector','ApprovalDate','ApprovalFY','Term','NumEmp','NewBiz',
                  'CreateJob','RetainedJob','Franchise','UrbanRural','LowDocu','DisburseDate','DisburseGross','GrAppv','SBA_Appv','Default']]
    df_loan = fix_Date_Year(df_loan, 'ApprovalDate', 'ApprovalFY')
    df_loan = df_loan.dropna()
    df_loan.to_pickle('data/pickled_loan')   