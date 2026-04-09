#all  neccesary libraries
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve,classification_report

#visualization

sea.set_style('whitegrid')
pd.set_option('display.max_columns',None)
plt.rcParams['Figure.dpi']=100

file= '/home/punit/Documents/Ds/sci-kit/UCI_Credit_Card.csv'

try:
    df = pd.read_csv(file)

    df.rename(columns={"default.payment.next.month":'Deafult',"PAY_0":'Pay_1'},inplace=True)

    if 'ID' in df.columns:
        df.drop('ID',axis=1,inplace=True)
except Exception as e:
    print("Camt load file ",e)
    exit()




