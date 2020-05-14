![](images/SBA_loan_header.jpg)

# Table of Contents
1. [Background & Motivation](#background)
2. [Data](#data)
3. [Data EDA](#eda)
4. [Models](#model)
5. [Results](#result)


## 1. Background & Motivation <a name="background"></a>

The U.S. Small Business Administration (SBA) is a United States government agency that provides support to entrepreneurs and small businesses.  The agency was founded on July 30, 1953 with the primary goals of promoting and assisting small enterprises in the U.S.  In the United States, small businesses have been a primary source of innovation and job creation.  According to a [JP Morgan report](https://www.jpmorganchase.com/corporate/institute/small-business-economic.htm), small businesses accounted for over half (51.2%) of the net job creation in 2014, and 48% of US overall employment as of 2014.   

One way the SBA provides assistance to small businesses is to provide easier access to the capital market and funding.  The agency doesn't extend loans directly.  Instead, it offers a loan guarantee program that is designed to encourage banks to grant commercial loans to small businesses.  In this regard, SMA acts much like an insurance provider to mitigate the risks for banks by shouldering some of the default risks.  Because SBA only guarantees a portion of the entire loan amount, banks who grant the loan will incur some losses if a small business defaults on its SBA-guaranteed loan.  As such, banks still need to evaluate the loan application and decide if they should grant it or not. 

We are in the middle of the pandemic Covid-19.  It has brought sudden and significant threats not only to human health at the risk of overcrowding the medical system but also to normal economic operations when people stay at home to "flatten the curve".  Many small businesses are suffering.  On April 21, 2020, the U.S. Senate approved a $484 billion coronavirus relief package that would provide loans to distressed small businesses.  These loans are administered by commercial banks as the "Paycheck Protection Program (PPP) loans".  In this study, I would like to build models on historical SBA guaranteed loans.  My goal is to identify the key risk factors that differentiate high-risk loans (resulting in default) from low-risk loans (with full repayments). 

## 2. Data & EDA <a name="data"></a>

I found the dataset on kaggle.com, though it originally came from the U.S. SBA.  It contains historical data from 1968 to 2014 (899,164 observations in total).  The variable name, the data type and a brief description of each variable is found in the below table:


| Variable name     | Data type | Description of variable                               |
|-------------------|-----------|-------------------------------------------------------|
| LoanNr_ChkDgt     | Text      | Loan identifier - Primary key                         |
| Name              | Text      | Borrower name                                         |
| City              | Text      | Borrower city                                         |
| State             | Text      | Borrower state                                        |
| Zip               | Text      | Borrower zip code                                     |
| Bank              | Text      | Bank (lender) name                                    |
| BankState         | Text      | Bank (lender) state                                   |
| NAICS             | Text      | North American industry classification system code    |
| ApprovalDate      | Date/Time | Date SBA approved the loan                            |
| ApprovalFY        | Text      | Fiscal year of loan commitment                        |
| Term              | Integer   | Loan term in months                                   |
| NoEmp             | Integer   | Number of business employees                          |
| NewExist          | Text      | 1=Existing business, 2=New business                   |
| CreateJob         | Integer   | Number of jobs created                                |
| RetainedJob       | Integer   | Number of jobs retained                               |
| FranchiseCode     | Text      | Franchise code, (00000 or 00001) = No franchise       |
| UrbanRural        | Text      | 1=Urban, 2=rural, 0=undefined                         |
| RevLineCr         | Text      | Revolving line of credit: Y=Yes, N=No                 |
| LowDoc            | Text      | LowDoc Loan Program: Y=Yes, N=No                      |
| ChgOffDate        | Date/Time | The date when a loan is declared to be in default     |
| DisbursementDate  | Date/Time | Loan disbursement date                                |
| DisbursementGross | Currency  | Gross loan amount disburse                            |
| BalanceGross      | Currency  | Gross amount outstanding                              |
| MIS_Status        | Text      | Loan status: charged off = CHGOFF, Paid in full = PIF |
| ChgOffPrinGr      | Currency  | Charged-off amount                                    |
| GrAppv            | Currency  | Gross amount of loan approved by bank (lender)        |
| SBA_Appv          | Currency  | SBA's guaranteed amount of approved loan              |

The following actions are taken to transform the data:

* __NAICS__ (North American Industry Classification System):  This is a 2- through 6-digit hierachical classfication system used by the Federal agencies to classify business establishments for the purpose of statiscial collection, analysis, ana presentation of data.  The first two digits of NAICS represents the economic sector.  Even though the original data contains the 6-digit code (despite many missing values), efforts are taken to reduce the code to the first two digits, therefore summarizing the loans to the sector level.  The below table shows the description of the NAICS sectors, with more details found on the [U.S. Census page](https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2017).

   | Sector | Description                                                              |
   |--------|--------------------------------------------------------------------------|
   | 11     | Agriculture, Forestry, Fishing and Hunting                               |
   | 21     | Mining, Quarrying, and Oil and Gas Extraction                            |
   | 22     | Utilities                                                                |
   | 23     | Construction                                                             |
   | 31-33  | Manufacturing                                                            |
   | 42     | Wholesale Trade                                                          |
   | 44-45  | Retail Trade                                                             |
   | 48-49  | Transportation and Warehousing                                           |
   | 51     | Information                                                              |
   | 52     | Finance and Insurance                                                    |
   | 53     | Real Estate and Rental and Leasing                                       |
   | 54     | Professional, Scientific, and Technical Services                         |
   | 55     | Management of Companies and Enterprises                                  |
   | 56     | Administrative and Support and Waste Management and Remediation Services |
   | 61     | Educational Services                                                     |
   | 62     | Health Care and Social Assistance                                        |
   | 71     | Arts, Entertainment, and Recreation                                      |
   | 72     | Accommodation and Food Serivices                                         |
   | 81     | Other Services (except Public Administration)                            |
   | 92     | Public Administration                                                    |

* __ApprovalFY__: In the original data set, there are 18 observations with ApprovalFY='1976A'.  I am not able to find what the 'A' indicates.  I choose to convert these values to 1976, thereby collapsing these 18 data points to be with ApprovalFY='1976'.

* __Franchise__: The original data set has mixed entry for this variable.  Entries '0000' and '0001' indicate there is no franchise, and franchise codes are entered wherever franchises are involved.  I choose to convert this variable to a Boolean type with 0 indicating no franchise and 1 otherwise. 

* __NewExist__ (1=Existing Business, 2=New Business): I choose to convert this variable to a Boolean type variable __NewBiz__ with 0 indicating existing business and 1 new business. 

* __LowDoc__ (Y=Yes, N=No): SBA implemented a "LowDoc Loan" program where loans under $150,000 can be processed using just a one-page application.  This is done for the goal of processing more loans efficiently.  I choose to convert this variable to a Boolean type variable __LowDocu__ where original values of '0' or 'N' corresponds to 0 and original values of '1' or 'Y' corresponds to 1. 

* __MIS_Status__: This is the target variable that we are interested in.  I choose to convert the variable to a Boolean type variable __Default__ where 1=CHGOFF and 0=PIF. 

* __Currency_Variables__:  The original data variables with type = 'Currency' are formated in the accounting convention with "$" and "," separators. These are converted to the float data type. 

* __SBA Guaranteed Portion__:  The original dataset contains the gross loan amount and the SBA guaranteed amount.  I choose to divide the latter by the former to obtain a new variable __SBA_g__ which is the percent of loan that is guanranteed by SBA.  It is a variable with values between 0 and 1.

* __U.S. Unemployment Rate__:  The final variable which isn't contained in the original SBA loan dataset is the U.S. unemployment rate, a monthly data series obtained from from the U.S. [Bureau of Labor Statistics](https://data.bls.gov/timeseries/LNS14000000). It is introduced as a proxy for where the U.S. economy stands during an economic cycle.  The unemployment rate data is merged with the SBA loan data on the loan approval date.  The thought behind this is loans issued during bad economic times (with high unemployment rate) can have higher default risk and vice versa. Below is a plot of the unemployment rate during 1965.01 to 2020.04.  Admittedly our SBA loan data ends roughly at mid 2014.

![](images/unemployment.png)

## 3. Data EDA <a name="eda"></a>

Which explanatory variables may be good predicators of whether a loan will go into default or not?  This section uses data exploratory analysis to investigate and gather useful features for the predicative model.  

### 3.1. Location (State)

Location (in this case State) can be a possible indicator for potential loan risk.  The 50 states and Washington DC have different economic environments and industry concentration, therefore leading to different default rates.  As shown by the heat map below, loan default rates definitely vary across states with Florida leading the pack at around 27.4%.  This could be explained by Florida suffering more from the boom and bust of real estate cycles, as well as more natural disaters like typhoons.  In contrast, western states like Montana and Wyoming have lower loan default rates (in the 6% range).  This could be because these states traditionally relies on the oil and mineral industries which were more or less stable during the sample periods.     

![](images/default_state.png)

### 3.2. Sectors

The below bar charts shows the top 10 highest loan default sectors.  Cross-referencing the sector NAICS table above, we can see that __sector 53__ is "Real estate and rental and leasing".  Real estate secotor is pro-cyclical and tends to overshoot in both economic booms and busts.  Since 1980, the United States has experienced at least two big real-estate led economic cycles, the most recent being the 2008-2009 financial crisis.  On the other of the spectrum, __sector 11__ (Agriculture, forestry, fishing and hunting), __sector 21__ (Mining, quarring, and oil and gas extraction), __sector 55__ (Management), and __sector 62__ (Health care and social assistance) suffer from lower loan default risks (in the 10% range) either because the sector is acyclical or hasn't experienced significant downward price adjustments.  (The same cannot be said about oil and gas industry in the Covid-19 Crisis, though the cause is not the virus alone.) 

![](images/default_sector.png)

### 3.3. Bank (Lender)

Banking is an industry with high regulations.  Typically commercial banks have long established guidance to assess loan risks.  Therefore we wouldn't expect the commercial bank or which state the bank reside in has any impact on the loan default rates, unless the bank is not following the regulatory rules.  Since there are many banking/credit union entities in the data set, most of issue less than 1000 loans.  I choose to plot two histograms of all banks extending 1000+ or 3000+ loans, see below charts.  As expected, we can't make the association that banks with more loans issued have lower / higher default rates.   

![](images/default_bank_3000.png)

![](images/default_bank_1000.png)

### 3.4. Business Characteristics

SBA dataset contains some business characteristics that might be influential in the performance of loans.  Specifically, 
* New business? - the assumption is established business have a proven record of success with established customers, operation workflow, etc.  In contrast new businesses unproven and might not weather economic ups and downs as well.  However, calculated data indicates otherwise:

|              | New Biz | Established Biz |
|--------------|--------:|----------------:|
| Default Rate |  18.60% |          17.11% |

* Franchised? - similar to the new business, one can argue that business with a franchised model might have a lower risk to default on loans.  This is because franchised businesses tend to have established business plans, target clientiles, smoothed supply chains, etc.  Yet the calculated data indicates otherwise:

|              | Franchise | Non-Franchise |
|--------------|----------:|--------------:|
| Default Rate |    15.16% |        17.68% |

* Number of Employees? - below is a quartile table. It seems that Non-default loans have more employees.  Perhaps bigger in size? 

| Quartiles    | Default | Non-Default |
|--------------|--------:|------------:|
| 100% maximum |    9999 |        9999 |
| 75% quartile |       7 |          11 |
| 50% median   |       3 |           5 |
| 25% quartile |       2 |           2 |
| Minimum      |       0 |           0 |

### 3.5. Loan Characteristics

Other variables relate more to the loan characteristics. Below are some comparisons of the quantiles for loans ending in default v.s. non-default. 

__Loan Gross Disbursement__
| Quartiles    |    Default | Non-Default |
|--------------|-----------:|------------:|
| 100% maximum | $4,362,157 | $11,446,325 |
| 75% quartile |   $140,417 |    $256,686 |
| 50% median   |    $62,000 |    $100,000 |
| 25% quartile |    $28,055 |     $49,000 |
| Minimum      |     $4,000 |      $4,000 |

__Loan Term (months)__
| Quartiles    | Default | Non-Default |
|--------------|--------:|------------:|
| 100% maximum |     461 | $11,446,325 |
| 75% quartile |      68 |         180 |
| 50% median   |      49 |          84 |
| 25% quartile |      29 |          78 |
| Minimum      |       0 |           0 |

__SBA Guarantee Portion (%)__
| Quartiles    | Default | Non-Default |
|--------------|--------:|------------:|
| 100% maximum |     100 |         100 |
| 75% quartile |      82 |          85 |
| 50% median   |      50 |          75 |
| 25% quartile |      50 |          50 |
| Minimum      |      12 |         2.8 |

__Default_by_LowDocu__
|              | LowDocu   | Non-LowDocu   |
|--------------|----------:|--------------:|
| Default Rate |    9.00%  |        18.74% |

Based on these numbers, we can draw initial conclusions that loans with higher gross disbursement dollar amounts, with longer loan terms and higher SBA guarantee ratios tend to have lower default risk.  This could be explained by that businesses getting these types of loans are more established, bigger in operation sizes, or have other collaterals posted(e.g. real estates).

With the initial data EDA, I made the choice to select the following variables to model loan default risk: 
* StateRisk
* SectorRisk
* LoanTerm(Term)
* Number of Employees(NumEmp)
* LowDocu
* Gross_Approval_Amt(GrAppv)
* SBA_Guaranteed_Ratio(SBA_g)
* Unemployment(U_rate)  

For StateRisk and SectorRisk, additional bucketing is done to convert the variables into numeric values.  For StateRisk, any state with a default rate higher than 18%(inclusive) is categorized as high risk (value=2). States with default rate between 10% and 18%(not inclusive) are rated 1 and those with default rate lower than 10% are rated 0.  For SectorRisk, ratings are similarly chosen amongst 0, 1, and 2, with the cutoff levels at default rate equal to 10% or 20%. 

## Models & Comparison <a name="model"></a>


## Results <a name="result"></a>