import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Data cleaning and preparing

data = pd.read_excel("/storage/emulated/0/documents/student_success_v2-1.xlsx",sheet_name=None)

df = data['Sheet1']

print("Information about the data")
print(df.info())
print("")
print("Descriptive statistics of the data")
print(df.describe())
print("")
print("No. of null points in each column")
print(df.isnull().sum())
print("")
print("Name of the columns in the data")
print(df.columns)
print("")

le = LabelEncoder()
df['Internet'] = le.fit_transform(df['Internet'])
df['Pass/Fail'] = le.fit_transform(df['Pass/Fail'])

Features = ['Study hours','Sleep hours','Attendance','Internet']
scaler = StandardScaler()
df_copy = df.copy()
df_copy[Features] = scaler.fit_transform(df_copy[Features])

X = df_copy[Features]
y = df_copy['Pass/Fail']

# Splitting testing and training data

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y
)

# Definig our model and training it

model = LogisticRegression()
model.fit(X_train,y_train)
print("The classification model is ",model)

predictions = model.predict(X_test)

# Checking performance of our model on the test data

print("")
print("Performance of the classifier")
print("")
print("Classification report:")
print(metrics.classification_report(y_test,predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test,predictions))
print("accuracy = ",accuracy_score(y_test, predictions)*100,"%")

# Making predictions using our model

print("")
print("----Student Success Predictor----")

Study_hours = int(input("Study hours:"))
Sleep_hours = int(input("Sleep hours:"))
Attendence = int(input("Attendance:"))
internet = input("Do you use internet: ")

if internet == 'yes':
    inte = 1
else:
    inte = 0
        
user_input = {        
    "Study hours":Study_hours,
    "Sleep hours":Sleep_hours,
    "Attendance":Attendence,
    "Internet":inte
}
user_input_df = pd.DataFrame(user_input,index=[0])
Scaled_array = scaler.transform(user_input_df)
Scaled_df = pd.DataFrame(Scaled_array,columns=user_input_df.columns)
prediction = model.predict(Scaled_df)
result = 'Pass' if prediction==1 else "Fail"
print("The student is likely to", result)

# Saving the model using joblib

file_name = 'Student_success_predicter.pkl'
joblib.dump(model, file_name)
print("")
print("Model successfully saved as", file_name)

# Loading our model

loaded_model = joblib.load(file_name)
print("")
print("The loaded model is",loaded_model)

# Making predictions using our loaded model

df_loaded_model = pd.DataFrame({
    "Study hours":[2,8,12,9.1],
    "Sleep hours":[5,3,2.5,3],
    "Attendance":[56,78,94,56],
    "Internet":[1,0,0,0]
})

array_loaded_model_scaled = scaler.transform(df_loaded_model)

df_loaded_model_scaled = pd.DataFrame(array_loaded_model_scaled, columns=df_loaded_model.columns)

pred = loaded_model.predict(df_loaded_model_scaled)

print("Prediction of our loaded model : ",pred)
