from BayesNet import *
import random
import pandas as pd
import numpy as np

random.seed("aima-python")

T, F = True, False

file = r"sleeping-alone-data.xls"
df = pd.read_excel(file)
print(df)

print("number of rows",df.shape[0])
print("number of columns",df.shape[1])
print("name of columns",df.columns)

#get domain of a single column values
print("Domain Gender",df['Gender'].unique())
print("Domain Age",df['Age'].unique())
print("Domain Education",df['Education'].unique())
print("Domain OftenSleepSeparateBeds",df['OftenSleepSeparateBeds'].unique())
print("Domain SeparateBedsStayTogether",df['SeparateBedsStayTogether'].unique())

#### Female
####   |
####   |
####  SeparateBedsStayTogether


# SeparateBedsStayTogether? and Female?

# num female
dfFemale = df.query('Gender==\'Female\'')
# print(dfFemale.shape[0])
percentFemale = dfFemale.shape[0] / df.shape[0]

# num not female
dfNotFemale = df.query('Gender!=\'Female\'')
# print(dfFemale.shape[0])
percentNotFemale = dfNotFemale.shape[0] / df.shape[0]


# female
dfSepBedsStayTogFemale = df.query('SeparateBedsStayTogether==\'Strongly agree\' or SeparateBedsStayTogether==\'Somewhat agree\' and Gender==\'female\'')
# print(dfSepBedsStayTogFemale)
# P(SepBedsStayTog|Female)
percentSepBedsStayTogFemale = (dfSepBedsStayTogFemale.shape[0]/df.shape[0])/(dfFemale.shape[0]/df.shape[0])

dfSepBedsStayTogNotFemale = df.query('SeparateBedsStayTogether==\'Strongly agree\' or SeparateBedsStayTogether==\'Somewhat agree\' and Gender!=\'female\'')
# print(dfSepBedsStayTogNotFemale)
# P(SepBedsStayTog|Not Female)
percentSepBedsStayTogNotFemale = (dfSepBedsStayTogNotFemale.shape[0]/df.shape[0])/(dfNotFemale.shape[0]/df.shape[0])

#### Female
####   |
####   |
####  SeparateBedsStayTogether

FemaleSeparateBedsStayTogether = BayesNet([('Female', '', percentFemale),
                                   ('SepBedsStayTog', 'Female',
                      {(T): percentSepBedsStayTogFemale, (F): percentSepBedsStayTogNotFemale})
                                  ])
print("SepBedsStayTog? and Female?")

#Bayes
print ("Given Female=T", enumeration_ask(
    'SepBedsStayTog', dict(Female=T),
    FemaleSeparateBedsStayTogether).show_approx())

print ("Given Female=F", enumeration_ask(
    'SepBedsStayTog', dict(Female=F),
    FemaleSeparateBedsStayTogether).show_approx())
