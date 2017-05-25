#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#import csv files
ba = pd.read_csv("../assets/bill_amount.csv",)
bid = pd.read_csv("../assets/bill_id.csv")
cd = pd.read_csv("../assets/clinical_data.csv")
demo = pd.read_csv("../assets/demographics.csv")

#check null values and datatypes for bill_amount and bill_id
ba.isnull().sum()/ba.shape[0]
ba.dtypes
bid.isnull().sum()/bid.shape[0]
bid.dtypes

#change date_of_admission to datetime
bid.date_of_admission = pd.to_datetime(bid.date_of_admission)

#check null values and datatypes for clinical data
cd.isnull().sum()/cd.shape[0]
cd.dtypes

cd.date_of_admission = pd.to_datetime(cd.date_of_admission)
cd.date_of_discharge = pd.to_datetime(cd.date_of_discharge)

#check values for medical history columns
cd.medical_history_3.value_counts()
cd.medical_history_2.value_counts()
cd.medical_history_5.value_counts()

#change "No" and "Yes" to binary
cd.medical_history_3 = cd.medical_history_3.map({"No":0, "Yes":1, "0":0, "1":1})

#drop null values
cd.dropna(inplace=True)
#change type to int
cd.medical_history_2 = cd.medical_history_2.astype('int')
cd.medical_history_5 = cd.medical_history_5.astype('int')

#check null values and datatypes for demo data
demo.isnull().sum()/demo.shape[0]
demo.dtypes

#change date of birth to datetime
demo.date_of_birth = pd.to_datetime(demo.date_of_birth)

#check values for gender, resident status and race
demo.gender.value_counts()
demo.resident_status.value_counts()
demo.race.value_counts()

#fix values
demo.gender = demo.gender.apply(lambda x: 0 if x=="Female" or x=="f" else 1)
demo.resident_status = demo.resident_status.apply(lambda x: "Singaporean" if x=="Singapore citizen" else x)
demo.race = demo.race.apply(lambda x: "Chinese" if x=="chinese" else "Indian" if x=="India" else x)

#merge bill amount and bill id dataframes on bill id
bill_df = ba.merge(bid,on="bill_id").copy(deep=True)

#change id to patient id to merge clinical data and demo data
cd.rename(columns={'id':'patient_id'}, inplace=True)

patient_df = cd.merge(demo,on="patient_id").copy(deep=True)

#let's make an age column from the birthdate
patient_df["age"] = pd.datetime.now().date() - patient_df["date_of_birth"]
#change timedelta dtype to years (int)
patient_df.age = patient_df["age"].apply(lambda x: x.days/365)

#make total_days from date of admission and discharge
patient_df["total_days"] = patient_df.apply(lambda x: (x["date_of_discharge"] - x["date_of_admission"]).days,axis=1)

#count of race by resident type
plt.figure(figsize=(8,6))
ax = sns.countplot(x="race", data=patient_df, hue="resident_status",palette="colorblind")

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/len(patient_df)), (x.mean(), y),
                ha='center', va='bottom') # set the alignment of the text

df = bill_df.merge(patient_df,on=["patient_id","date_of_admission"])

#group bill amounts of the same patient on the same day
grp_df = pd.DataFrame(df.groupby(["patient_id","date_of_admission"]).amount.sum())
grp_df.reset_index(level=0,inplace=True)
grp_df["date_of_admission"] = grp_df.index

#merge to df and drop initial amount and bill_id
df = df.merge(grp_df,on=["patient_id","date_of_admission"]).copy(deep=True)
df.drop(["amount_x","bill_id"],axis=1,inplace=True)

#drop duplicates
df.drop_duplicates(inplace=True)
df.shape

#distribution amount
sns.distplot(df.amount_y)

#describe
df.amount_y.describe()

#analyze the demographic data and the bill amount
df.groupby(["race","gender"]).amount_y.sum()
df.groupby(["race","resident_status"]).amount_y.sum()

sns.boxplot(x="resident_status", y="amount_y", hue="gender", data=df, palette="PRGn")
sns.boxplot(x="race", y="amount_y", hue="resident_status", data=df, palette="colorblind")

#check yearly admission
df.groupby(df['date_of_admission'].map(lambda x: x.year))["resident_status"].value_counts()

#get average amount per day in hospital
df["avg_bill"] = df["amount_y"]/df["total_days"]
df.groupby("race").avg_bill.median()

sns.distplot(df.avg_bill)
np.median(df.avg_bill)

#look at the age distribution
df.age.describe()
sns.distplot(df.age)

#check correlation of variables
sns.heatmap(df.corr())

#check distribution of lab_results
sns.distplot(df.lab_result_1)
sns.distplot(df.lab_result_2)
sns.distplot(df.lab_result_3)

#check baselines for medical history and symptoms
df.medical_history_1.value_counts()/len(df)
df.medical_history_2.value_counts()/len(df)
df.medical_history_3.value_counts()/len(df)
df.medical_history_4.value_counts()/len(df)
df.medical_history_5.value_counts()/len(df)
df.medical_history_6.value_counts()/len(df)
df.medical_history_7.value_counts()/len(df)

df.symptom_1.value_counts()/len(df)
df.symptom_2.value_counts()/len(df)
df.symptom_3.value_counts()/len(df)
df.symptom_4.value_counts()/len(df)
df.symptom_5.value_counts()/len(df)

#check distribution of total days
sns.distplot(df.total_days)
np.mean(df.total_days)

#column to show how many disease a patient has (medical history)
df["total_mh"] = df["medical_history_1"] + df["medical_history_2"] + df["medical_history_3"] + df["medical_history_4"] + df["medical_history_5"]+ df["medical_history_6"] + df["medical_history_7"]
sns.distplot(df.total_mh)

#column to show how many symptoms a patient has
df["total_sym"] = df["symptom_1"] + df["symptom_2"] + df["symptom_3"] + df["symptom_4"] + df["symptom_5"]
sns.distplot(df.total_sym)

#distribution of weight and height
sns.distplot(df.weight)
sns.distplot(df.height)

#import for modeling
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score

def regressionScore(model, name, myX,myy):
    s = cross_val_score(model, myX, myy, cv=5, scoring="r2")
    print "{} Score:\t{:0.3} ± {:0.3}".format(name, s.mean().round(3), s.std().round(3))

def classifierScore(model, name, myX, myy):
    s = cross_val_score(model, myX, myy, cv=5, scoring="accuracy")
    print "{} Score:\t{:0.3} ± {:0.3}".format(name, s.mean().round(3), s.std().round(3))

def fitAndPrint1(model,X,y):
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    ## The line / model
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    
    print "Score:", model.score(X_test, y_test)

def fitAndPrint(model,X,y):
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=random_state)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print "Accuracy Score: %s"%(accuracy_score(y_test,y_pred))

#define random state
random_state = np.random.RandomState(0)

#enconde race and resident_status
df["race"] = LabelEncoder().fit_transform(df["race"].values)
df["resident_status"] = LabelEncoder().fit_transform(df["resident_status"].values)

#initial list of features to estimate bill amount
xlist = ["medical_history_1","medical_history_2","medical_history_3","medical_history_4",
         "medical_history_5","medical_history_6","medical_history_7","symptom_1","symptom_2",
         "symptom_3","symptom_4","symptom_5","weight","height","gender","resident_status",
         "race","age"]

X = df[xlist]
y = df["amount_y"]

#instantiate linear regression
lm = LinearRegression()

#checks R2 score using cross validation
regressionScore(lm,"Linear Regression",X,y)

#instantiate random forest and check cross val score
rf = RandomForestRegressor()
regressionScore(rf,"Random Forest Regressor",X,y)

#higher rf score, so fit and print values
fitAndPrint1(rf,X,y)

#use random forest feature importances to reduce our features
filteredlist = []
for x,y in zip(rf.feature_importances_, xlist):
    if x> 0.01:
        filteredlist.append(y)

X = df[filteredlist]
y = df["amount_y"]

score(rf,"Random Forest",X,y)

#cross val score higher than initial score with less features - less processing time








