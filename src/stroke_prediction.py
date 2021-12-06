import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics, preprocessing
#------------------------------------------------------------------------------------------

dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = pd.DataFrame(dataset)

# ____________________________________________ SUBQUESTION - A - ____________________________________________#
# ~~~~~~~~~~~~~~~~~~~~~~~~GRAPHS~~~~~~~~~~~~~~~~~~~~~~~~#
'''
#GENDER
print("Gender\n",df['gender'].value_counts())
#df['gender'].value_counts()[:].plot(kind='bar')
sns.countplot(x = "gender", data = df, order = df['gender'].unique())
plt.xlabel('Gander')
plt.ylabel('Number of patients')
plt.show() # plot of gender

x,y = 'gender', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Gander')
plt.ylabel('Percent of Stroke')
plt.show()

#AGE
conditions = [
    (df['age'] <= 10),
    (df['age'] > 10) & (df['age'] <= 20),
    (df['age'] > 20) & (df['age'] <= 30),
    (df['age'] > 30) & (df['age'] <= 40),
    (df['age'] > 40) & (df['age'] <= 50),
    (df['age'] > 50) & (df['age'] <= 60),
    (df['age'] > 60) & (df['age'] <= 70),
    (df['age'] > 70) & (df['age'] <= 80),
    (df['age'] > 80)
    ]
# create a list of the values we want to assign for each condition
values = ['0-10', '11-20', '21-30', '31-40','41-50','51-60','61-70','71-80','>80']
# display updated DataFrame
df['age_categories'] = np.select(conditions, values) # adding column age_categories
print("Age\n", df['age_categories'].value_counts(ascending=False))
sns.countplot(x = "age_categories", data = df, order = values)
plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.show() # plot of age

x,y = 'age_categories', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Age')
plt.ylabel('Percent of Stroke')
plt.show()


df.pop('age_categories') #drop column age_categories


#HYPERTENSION
print("Hypertension\n",df['hypertension'].value_counts())
sns.countplot(x = "hypertension", data = df, order = df['hypertension'].unique())
plt.xlabel('Hypertension')
plt.ylabel('Number of patients')
plt.show() # plot of hypertension

x,y = 'hypertension', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('hypertension')
plt.ylabel('Percent of Stroke')
plt.show()


#HEART DISEASE
print("Heart Disease\n",df['heart_disease'].value_counts())
sns.countplot(x = "heart_disease", data = df, order = df['heart_disease'].unique())
plt.xlabel('Heart Disease')
plt.ylabel('Number of patients')
plt.show() # plot of heart_disease

x,y = 'heart_disease', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Heart Disease')
plt.ylabel('Percent of Stroke')
plt.show()


#Ever Married
print("Ever Married\n",df['ever_married'].value_counts())
sns.countplot(x = "ever_married", data = df, order = df['ever_married'].unique())
plt.xlabel('Ever Married')
plt.ylabel('Number of patients')
plt.show() # plot of ever_married

x,y = 'ever_married', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Ever Married')
plt.ylabel('Percent of Stroke')
plt.show()


#WORK TYPE
print("Work Type\n",df['work_type'].value_counts())
sns.countplot(x = "work_type", data = df, order = df['work_type'].unique())
plt.xlabel('Work Type')
plt.ylabel('Number of patients')
plt.show() # plot of work_type

x,y = 'work_type', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Work Type')
plt.ylabel('Percent of Stroke')
plt.show()


#RESIDENCE TYPE
print("Residence Type\n",df['Residence_type'].value_counts())
sns.countplot(x = "Residence_type", data = df, order = df['Residence_type'].unique())
plt.xlabel('Residence Type')
plt.ylabel('Number of patients')
plt.show() # plot of Residence_type

x,y = 'Residence_type', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Residence type')
plt.ylabel('Percent of Stroke')
plt.show()


#AVERAGE GLUCOSE LEVEL
conditions = [
    (df['avg_glucose_level'] <= 60),
    (df['avg_glucose_level'] > 60) & (df['avg_glucose_level'] <= 80),
    (df['avg_glucose_level'] > 80) & (df['avg_glucose_level'] <= 100),
    (df['avg_glucose_level'] > 100) & (df['avg_glucose_level'] <= 120),
    (df['avg_glucose_level'] > 120) & (df['avg_glucose_level'] <= 140),
    (df['avg_glucose_level'] > 140) & (df['avg_glucose_level'] <= 160),
    (df['avg_glucose_level'] > 160) & (df['avg_glucose_level'] <= 180),
    (df['avg_glucose_level'] > 180) & (df['avg_glucose_level'] <= 200),
    (df['avg_glucose_level'] > 200)
    ]
# create a list of the values we want to assign for each condition
values = ['0-60','60-80','80-100','100-120','120-140','140-160','160-180','180-200','>200']
df['avg_glucose_level_categories'] = np.select(conditions, values) # adding column avg_glucose_level_categories
print("Avg Glucose Level\n", df['avg_glucose_level_categories'].value_counts(ascending=False))
sns.countplot(x = "avg_glucose_level_categories", data = df, order = values)
plt.xlabel('Avg Glucose Level')
plt.ylabel('Number of patients')
plt.show() # plot of avg_glucose_level_categories

x,y = 'avg_glucose_level_categories', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Avg Glucose Level')
plt.ylabel('Percent of Stroke')
plt.show()

df.pop('avg_glucose_level_categories') #drop column avg_glucose_level_categories


# BMI
conditions = [ # to N/A ????????
    (df['bmi'] <= 10.0),
    (df['bmi'] > 10.0) & (df['bmi'] <= 15.0),
    (df['bmi'] > 15.0) & (df['bmi'] <= 20.0),
    (df['bmi'] > 20.0) & (df['bmi'] <= 25.0),
    (df['bmi'] > 25.0) & (df['bmi'] <= 30.0),
    (df['bmi'] > 30.0) & (df['bmi'] <= 35.0),
    (df['bmi'] > 35.0) & (df['bmi'] <= 40.0),
    (df['bmi'] > 40.0) & (df['bmi'] <= 45.0),
    (df['bmi'] > 45.0) & (df['bmi'] <= 50.0),
    (df['bmi'] > 50.0)
]
# create a list of the values we want to assign for each condition
values = ['<10', '10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50', '>50']
df['bmi_categories'] = np.select(conditions, values)  # adding column avg_glucose_level_categories
print("BMI\n", df['bmi'].value_counts(ascending=False))
sns.countplot(x="bmi_categories", data=df, order=values)
plt.xlabel('BMI')
plt.ylabel('Number of patients')
plt.show()  # plot of bmi

x,y = 'bmi_categories', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('BMI')
plt.ylabel('Percent of Stroke')
plt.show()

df.pop('bmi_categories')  # drop column bmi_categories


#SMOKING STATUS
print("Smoking Status\n",df['smoking_status'].value_counts())
sns.countplot(x = "smoking_status", data = df, order = df['smoking_status'].unique())
plt.xlabel('Smoking Status')
plt.ylabel('Number of patients')
plt.show() # plot of smoking_status

x,y = 'smoking_status', 'stroke'
(df
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xlabel('Smoking Status')
plt.ylabel('Percent of Stroke')
plt.show()


#STROKE
print("Stroke\n",df['stroke'].value_counts())
sns.countplot(x = "stroke", data = df, order = df['stroke'].unique())
plt.xlabel('Stroke')
plt.ylabel('Number of patients')
plt.show() # plot of stroke

# ~~~~~~~~~~~~~~~~~~~~~~~~ END of GRAPHS ~~~~~~~~~~~~~~~~~~~~~~~~#
# _________________________________________ END OF SUBQUESTION - A - _________________________________________#
'''


# ____________________________________________ SUBQUESTION - B - ____________________________________________#
smaller_df = df.copy()

# ENCODING
#--------------------------------------------------------------------------------
df_encoded = pd.get_dummies(smaller_df["gender"]) #gender - categorical variable - encoding
df_encoded= pd.concat([smaller_df,df_encoded], axis=1)
df_encoded.drop(["gender","Other"],inplace=True, axis=1)
df_encoded=df_encoded.rename(columns={"Female": "gender_female", "Male": "gender_male"})
smaller_df=df_encoded.copy()
#--------------------------------------------------------------------------------
df_encoded = pd.get_dummies(smaller_df["ever_married"])#ever_married - categorical variable - encoding
df_encoded= pd.concat([smaller_df,df_encoded], axis=1)
df_encoded.drop(["ever_married","No"],inplace=True, axis=1)
df_encoded=df_encoded.rename(columns={"Yes": "ever_married_yes"})
smaller_df=df_encoded.copy()
#---------------------------------------------------------------------------------
df_encoded = pd.get_dummies(smaller_df["work_type"])#work_type - categorical variable - encoding
df_encoded= pd.concat([smaller_df,df_encoded], axis=1)
df_encoded.drop(["work_type","children"],inplace=True, axis=1)
df_encoded=df_encoded.rename(columns={"Self-employed": "wt_self-employed", "Private": "wt_private","Govt_job":"wk_govt_job","Never_worked":"wt_never_worked"})
smaller_df=df_encoded.copy()
#---------------------------------------------------------------------------------
df_encoded = pd.get_dummies(smaller_df["Residence_type"])#Residence_type - categorical variable - encoding
df_encoded= pd.concat([smaller_df,df_encoded], axis=1)
df_encoded.drop(["Residence_type","Rural"],inplace=True, axis=1)
df_encoded=df_encoded.rename(columns={"Urban": "rt_urban"})
# dataframe has been encoded for all the categorical except smoking status
smaller_df=df_encoded.copy() #encoded dataframe for drop column method
df_copy2=smaller_df.copy() #encoded dataframe for averages
df_copy3=smaller_df.copy() #encoded dataframe for linear regression for bmi
df_copy4=smaller_df.copy() #encoded dataframe for linear regression for smoking status

# ~~~~~~~~~~~~~~~~~~~~~~~~DROP COLUMN~~~~~~~~~~~~~~~~~~~~~~~~#
#dropping columns method
smaller_df.drop('bmi', inplace=True, axis=1)  # inplace=True means the operation would work on the original object & axis=1 means we are dropping the column, not the row.
smaller_df.drop('smoking_status', inplace=True, axis=1)
print(smaller_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~ END of DROP COLUMN ~~~~~~~~~~~~~~~~~~~~~~~~#

# ~~~~~~~~~~~~~~~~~~~~~~~~AVERAGE~~~~~~~~~~~~~~~~~~~~~~~~#

# mean from pandas -> missing values (einai etoimo)
bmi_mean = round(df_copy2['bmi'].mean(axis=0, skipna=True), 2)
#print(bmi_mean)
df_copy2["bmi"].replace({np.nan: bmi_mean}, inplace=True) # replace N/A with bmi's mean
#print(df)

smoking = (df_copy2.groupby(['smoking_status']).count())['id']
#print(smoking)
smoking_mean = smoking.idxmax()
df_copy2["smoking_status"].replace({'Unknown': smoking_mean}, inplace=True) # replace Unknown with smoking's most common

# ~~~~~~~~~~~~~~~~~~~~~~~~ END of AVERAGE ~~~~~~~~~~~~~~~~~~~~~~~~#

# ~~~~~~~~~~~~~~~~~~~~~~~~LINEAR REGRESSION~~~~~~~~~~~~~~~~~~~~~~~~#

df_copy3.pop("smoking_status") #dropping smoking_status, we don't want it

#print(df_copy3.columns)
nanrows=df_copy3[df_copy3['bmi'].isna()] #getting all the rows with bmi==N/A
nanrows_knn=nanrows.copy() # for knn
X_pred_bmi=nanrows[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'stroke', 'gender_female', 'gender_male', 'ever_married_yes',
       'wk_govt_job', 'wt_never_worked', 'wt_private', 'wt_self-employed',
       'rt_urban']]
#print(X_pred)
#print("NANAROWS\n",nanrows)

df_copy3=df_copy3.dropna(subset=['bmi'], thresh=1) #dropping rows with bmi==N/A
df_copy5=df_copy3.copy() # for knn
#print(df_copy3)
X_tr_bmi=df_copy3[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'stroke', 'gender_female', 'gender_male', 'ever_married_yes',
       'wk_govt_job', 'wt_never_worked', 'wt_private', 'wt_self-employed',
       'rt_urban']]
Y_tr_bmi=df_copy3["bmi"]
#print(X.shape,Y.shape)

#---------------------------------------------------------------------------------
# LINEAR REGRESSION PREDICTION
model = LinearRegression()
model.fit(X_tr_bmi,Y_tr_bmi)
Y_pred = model.predict(X_pred_bmi)
print("PREDICTION FOR BMI:\n",Y_pred)
#Y_pred = pd.DataFrame(Y_pred, columns = ["bmi"])
nanrows["bmi"]=Y_pred # replace nan with predicted bmi
print("BMI PREDICT",nanrows["bmi"])
df_copy3= pd.concat([df_copy3,nanrows], axis=0) # COMPLETE DATAFRAME with predicted values
print("BMI\n",df_copy3)
#print(df_copy3["bmi"])
print("LR-BMI",df_copy3.columns)
print("END OF LINEAR REGRESSION FOR BMI\n========================================================")

#======================================================================================================================

df_copy4.pop("bmi") #dropping bmi, we don't want it

le= LabelEncoder()
df_copy4["smoking_status"]=le.fit_transform(df_copy4["smoking_status"])
print(df_copy4["smoking_status"]) # 0 -> Unknown
#---------------------------------------------------------------------------------
print(df_copy4.columns)


unknown_rows=df_copy4[df_copy4['smoking_status']==0] #getting all the rows with smoking_status==Unknown
unknown_rows_knn=unknown_rows.copy() # for knn

X_pred=unknown_rows[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'stroke', 'gender_female', 'gender_male',
       'ever_married_yes', 'wk_govt_job', 'wt_never_worked', 'wt_private',
       'wt_self-employed', 'rt_urban']]

df_copy4.drop(df_copy4.loc[df_copy4['smoking_status']==0].index, inplace=True)#dropping rows with smoking_status==Unknown
df_copy6=df_copy4.copy() # for knn copy
X=df_copy4[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'stroke', 'gender_female', 'gender_male',
       'ever_married_yes', 'wk_govt_job', 'wt_never_worked', 'wt_private',
       'wt_self-employed', 'rt_urban']]
Y=df_copy4["smoking_status"]
#---------------------------------------------------------------------------------
# LINEAR REGRESSION PREDICTION
model = LinearRegression()
model.fit(X,Y)
Y_pred = model.predict(X_pred)
print("PREDICTION FOR SMOKING:\n",Y_pred)
Y_pred=Y_pred.round(0).astype('int')
print("After",Y_pred)
unknown_rows["smoking_status"]=Y_pred # replace unknown==0 with predicted smoking status
#print(unknown_rows["smoking_status"])
df_copy4= pd.concat([df_copy4,unknown_rows], axis=0) # COMPLETE DATAFRAME with predicted values
print(df_copy4)
#print(df_copy4["smoking_status"])
print("LR-SMOK",df_copy4.columns)
print("END OF LINEAR REGRESSION FOR SMOKING STATUS\n========================================================")

# ~~~~~~~~~~~~~~~~~~~~~~~~ END of LINEAR REGRESSION ~~~~~~~~~~~~~~~~~~~~~~~~#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~KNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def find_K(x_tr, y_tr, k):
       print('\nStarting training for values of k', [each for each in k])
       knn = KNeighborsClassifier()# Creating an knn object
       parameters = {'n_neighbors': k} # =k neighbors list

       # Training the model
       model = GridSearchCV(knn, param_grid=parameters, cv=None) # cv=None, to use the default 5-fold cross validation
       model.fit(x_tr, y_tr)

       return model.best_params_["n_neighbors"]

possible_k=[]
for i in range(1,60):
       possible_k.append(i)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(Y_tr_bmi)
#finding suitable k
k=find_K(X_tr_bmi,encoded,possible_k) # Y_tr_bmi as agruement it had an error so I enconded
print(k," ",type(k))
# KNN - BMI==================================================================
#print(df_copy5)
knn = KNeighborsRegressor(n_neighbors=k) # Create the knn model
knn.fit(X_tr_bmi, Y_tr_bmi)# Fit the model on the training data
knn_pred = knn.predict(X_pred_bmi)# Make point predictions on the test set using the fit model
print("After KNN prediction for bmi's nan\n",knn_pred)
nanrows_knn["bmi"]=knn_pred # replace unknown==0 with predicted smoking status
#print(unknown_rows["smoking_status"])
df_copy5= pd.concat([df_copy5,nanrows_knn], axis=0) # COMPLETE DATAFRAME with predicted values
#print(df_copy5)
#print(df_copy5["bmi"])
print("KNN-BMI",df_copy5.columns)


print("END OF KNN FOR BMI\n========================================================")
# KNN - SMOKING STATUS==================================================================
#finding suitable k
k=find_K(X,Y,possible_k)
print(k," ",type(k))
#print(df_copy6)
knn = KNeighborsRegressor(n_neighbors=k) # Create the knn model
knn.fit(X, Y)# Fit the model on the training data
knn_pred = knn.predict(X_pred)# Make point predictions on the test set using the fit model
knn_pred=knn_pred.round(0).astype('int')
print("After KNN prediction for smoking_status' unknown\n",knn_pred)
unknown_rows_knn["smoking_status"]=knn_pred # replace unknown==0 with predicted smoking status
#print(unknown_rows["smoking_status"])
df_copy6= pd.concat([df_copy6,unknown_rows_knn], axis=0) # COMPLETE DATAFRAME with predicted values
#print(df_copy6)
#print(df_copy6["smoking_status"])
print("END OF KNN FOR SMOKING STATUS\n========================================================")
print("KNN-Smoking",df_copy6.columns)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END of KNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# ____________________________________________ SUBQUESTION - C - ____________________________________________#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Random Forest~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

Y = smaller_df['stroke']
X=smaller_df[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'gender_female', 'gender_male', 'ever_married_yes',
       'wk_govt_job', 'wt_never_worked', 'wt_private', 'wt_self-employed',
       'rt_urban']]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets

rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("test_Y",test_Y)
print("For Drop Columns\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))
print("ABS error:",metrics.mean_absolute_error(test_Y, predictions))

#--------------------------------------------------------------------------------------

#FOR AVERAGE
le= LabelEncoder() # encoding smoking status
df_copy2["smoking_status"]=le.fit_transform(df_copy2["smoking_status"])

Y = df_copy2['stroke']
X=df_copy2[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'bmi', 'smoking_status', 'gender_female', 'gender_male',
       'ever_married_yes', 'wk_govt_job', 'wt_never_worked', 'wt_private',
       'wt_self-employed', 'rt_urban']]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets
rf = RandomForestRegressor(n_estimators = 30, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("For Average\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))
#--------------------------------------------------------------------------------------
# For Linear Regression - BMI
Y = df_copy3['stroke']
X=df_copy3[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'bmi', 'gender_female', 'gender_male', 'ever_married_yes',
       'wk_govt_job', 'wt_never_worked', 'wt_private', 'wt_self-employed',
       'rt_urban']]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets
rf = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("For Linear Regression - BMI\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))
#--------------------------------------------------------------------------------------
# For Linear Regression - SMOKING STATUS
Y = df_copy4['stroke']
X=df_copy4[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'smoking_status', 'gender_female', 'gender_male',
       'ever_married_yes', 'wk_govt_job', 'wt_never_worked', 'wt_private',
       'wt_self-employed', 'rt_urban']]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets
rf = RandomForestRegressor(n_estimators = 50, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("For Linear Regression - Smoking Status\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))
#--------------------------------------------------------------------------------------
# For KNN - BMI
Y = df_copy5['stroke']
X=df_copy5[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'bmi', 'gender_female', 'gender_male', 'ever_married_yes',
       'wk_govt_job', 'wt_never_worked', 'wt_private', 'wt_self-employed',
       'rt_urban']]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets
rf = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("For KNN - BMI\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))
#--------------------------------------------------------------------------------------
# For KNN - SMOKING STATUS
Y = df_copy6['stroke']
X=df_copy6[['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'smoking_status', 'gender_female', 'gender_male',
       'ever_married_yes', 'wk_govt_job', 'wt_never_worked', 'wt_private',
       'wt_self-employed', 'rt_urban']]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Spitting data into training and testing sets
rf = RandomForestRegressor(n_estimators = 30, random_state = 42)
rf.fit(train_X, train_Y)# Train the model on training data
predictions = rf.predict(test_X)
predictions=predictions.round(0).astype('int') # because stroke is 0 or 1.
print(predictions)
print("For KNN - Smoking Status\nF1 score:",f1_score(test_Y, predictions, average='macro'))
print("Precision:",precision_score(test_Y, predictions, average='macro'))
print("Recall:",recall_score(test_Y, predictions, average='macro'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END of Random Forest ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# _________________________________________ END OF SUBQUESTION - C - _________________________________________#
