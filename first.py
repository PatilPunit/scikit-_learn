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
plt.rcParams['figure.dpi']=100

file= '/home/punit/Documents/Ds/sci-kit/UCI_Credit_Card.csv'

try:
    df = pd.read_csv(file)

    df.rename(columns={"default.payment.next.month":'Default',"PAY_0":'PAY_1'},inplace=True)

    if 'ID' in df.columns:
        df.drop('ID',axis=1,inplace=True)
except Exception as e:
    print("Camt load file ",e)
    exit()

x = df.drop('Default',axis=1)
y=df['Default']

category_features=["SEX","EDUCATION","MARRIAGE","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",]
numerical_features=x.columns.drop(category_features).tolist()


print('\n','='*70)
print("Exploratery data analysis")
print('='*70)


print("Df info - Viewing Every columns info")
(df.info())

print("\n 2.Missing Value Imputations : ")
missing_value_sum=df.isnull().sum()

if missing_value_sum.max() > 0:
    print("missing Values .Imputing numerical features with mean")
    imputer = SimpleImputer(strategy='Mean')
    df[numerical_features]=imputer.fit_transform(df[numerical_features])
else :
    print("No missing vales in dataset")

default_counts=y.value_counts()
default_ratio = default_counts[1]/default_counts.sum()

print("Deafult count of 0 : ",default_counts[0])
print("default count of 1: ",default_counts[1])
print(f"default rate: {default_ratio:.2%}")

plt.figure(figsize=(6,4))
sea.countplot(x=y , order=[0,1])
plt.title=("default count 0 = No default 1 = Default")
plt.xlabel("Next month (0=no,1=yes)")
plt.ylabel('Count')
plt.show()
"""
print("Boxplot for numerical featues")
n_col=4
n_rows=int(np.ceil(len(numerical_features)/ n_col))
plt.figure(figsize=(16,4*n_rows))
for i,feat in enumerate(numerical_features):
   plt.subplot(n_rows,n_col,i+1)
   sea.boxplot(y=df[feat],orient='v')
   
   plt.ylabel(None)
plt.suptitle("box plot of numerical features")
plt.tight_layout(rect=[0,0.03,1,0.097])
plt.show()

print("Heatmap showing correaletion between thier features (Numerical only)")
correletion_matrix=df[numerical_features].corr()
plt.figure(figsize=(16,12))
mask = np.triu(correletion_matrix)
sea.heatmap(correletion_matrix,cmap='coolwarm', annot=True,fmt='.2f',mask = mask,cbar_kws={'shrink':0.8},annot_kws={'size':8})
# plt.title("heatmap of correletion of numerical features")
plt.show()

n_col=4
n_rows=int(np.ceil(len(numerical_features)/ n_col))
plt.figure(figsize=(16,4*n_rows))
for i,feat in enumerate(numerical_features):
   plt.subplot(n_rows,n_col,i+1)
   sea.histplot(data=df, hue='Default',x=feat,kde=True,bins=20,palette='viridis',common_norm=False)
   plt.ylabel(None)
plt.suptitle("box plot of numerical features")
plt.tight_layout(rect=[0,0.03,1,0.097])
plt.show()
print("Same analysis fir numerical features")
n_col=3
n_rows=int(np.ceil(len(category_features)/ n_col))
plt.figure(figsize=(16,5*n_rows))
for i,feat in enumerate(category_features):
   default_rate = df.groupby(feat) ['Default'].mean().sort_index()
   plt.subplot(n_rows,n_col,i+1)
   sea.barplot(x=default_rate.index,y=default_rate.values,color='skyblue')
   plt.ylabel('Default rate')
   plt.xlabel(feat)
   
plt.suptitle("box plot of numerical features")
plt.tight_layout(rect=[0,0.03,1,0.097])
plt.show()
"""
print("Manual representation and model implementation")
print("\n","="*70)
print("="*70)

#split data into training and testing sets

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
#manual processing (one hot coding)


x_train_encoded=pd.get_dummies(X_train,columns=category_features,drop_first=True)
x_test_encoded=pd.get_dummies(X_test,columns=category_features,drop_first=True)
train_cols=x_train_encoded.columns
x_test_encoded=x_test_encoded.reindex(columns=train_cols,fill_value=0)
numcols_to_scale=[col for col in train_cols if col in numerical_features]

#feature scaling
scaler = StandardScaler()
x_train_encoded[numcols_to_scale]=scaler.fit_transform(x_train_encoded[numcols_to_scale])
x_test_encoded[numcols_to_scale]=scaler.transform(x_test_encoded[numcols_to_scale])

print(" preprocesing complete")
print('initilize and train the model')

coustem_whieght={0:1.0,1:4.5}
model = LogisticRegression(

    solver = 'liblinear',
    class_weight='balanced',
    random_state=42,
    max_iter=1000

)
model.fit(x_train_encoded,Y_train)
print("Traing and testing complete")

print("10.prediction adn evaluation")

y_pred=model.predict(x_test_encoded)
y_proba=model.predict_proba(x_test_encoded)[:, 1]

print("Final model analysis")
print("\n","="*70)
print("="*70)

#confusion matrix

cm=confusion_matrix(Y_test,y_pred)
print("Prediction (Prediction vs actual)")
cm_df=pd.DataFrame(
    cm,
    index=['Actual No default(0)', 'Actual Default(1)'],
    columns=['Predicted No default(1)' ,'predicted default(1)']
)
print(cm_df)

roc_auc = roc_auc_score(Y_test,y_proba)
fpr,tpr,_=roc_curve(Y_test,y_proba)

plt.figure(figsize=(7,6))
plt.plot(fpr,tpr,color='Darkorange',lw=2,label=f"ROC Curve (area={roc_auc:.4%}")
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlabel("False default rate")
plt.ylabel("True positive rate")
plt.show()