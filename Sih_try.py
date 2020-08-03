import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

location_encoder=LabelEncoder()
Sector_encoder=LabelEncoder()
job_encoder=LabelEncoder()
Eligibility_encoder=CountVectorizer()


cv=CountVectorizer()
#cv.fit_transform(data["Eli"])

def Convert(df):## train and test set
    encod=df[["Location","Job_Description","Sector","salary","Month"]]
  #  encod=pd.DataFrame()
    encod["Location"]=location_encoder.fit_transform(df["Location"])
    encod["Job_Description"]=job_encoder.fit_transform(df["Job_Description"])
    encod["Sector"]=Sector_encoder.fit_transform(df["Sector"])
    trans=cv.fit_transform(df["Eligibility"])
    trans=pd.DataFrame(trans.todense(),columns=cv.get_feature_names())# Create a dataframe with only Eligibility values
    df2=pd.concat([encod,trans],axis=1)
    return df2

## pridicting data Set


def fun(salary1, job_label,Sector_label,City_label, Education, Month=0):
	
	salary = int(salary1)
	salary = 10*salary
	
	education_encoded=cv.transform([Education])
	
	job_int=job_encoder.transform([job_label])
	
	city_int=location_encoder.transform([City_label])
	
	sector_int=Sector_encoder.transform([Sector_label])
	
	df1=pd.DataFrame([[city_int[0],job_int[0],sector_int[0],salary,Month]],columns=["Location","Job_Description","Sector","salary","Month"])
	
	df2=pd.DataFrame(education_encoded.todense(),columns=cv.get_feature_names())
	
	df1=pd.concat([df1,df2],axis=1)
	topre2 = df1
	for i in range(11):
		df1 = pd.concat([df1,topre2],axis = 0)
	for i in range(12):
		df1.iloc[i,0] = i+1
	return df1
    
'''def Convert(df):
    enco = df[["Month","Location","Job_Description","Sector","salary","Eligibility"]]
    enco["City"]      = le1.fit_transform(enco["City"])
    enco["job_title"] = le2.fit_transform(enco["job_title"])
    enco["Sector"]    = le3.fit_transform(enco["Sector"])
    trans             = cv.fit_transform(enco["Eli"])
    trans             = pd.DataFrame(trans.todense(), columns=cv.get_feature_names())
    enco              = pd.concat([enco,trans],axis = 1)
    enco              = enco.drop(["Eli"],axis = 1)
    return enco'''
    
def make_data(salary,job,sector,city,Eli):
    month = 1;
    city1 = location_encoder.transform([city])[0]
    job1  = job_encoder.transform([job])[0]
    sec1  = Sector_encoder.transform([sector])[0]
    topre = pd.DataFrame([city1,job1,sec1,salary,month])
    topre = pd.DataFrame(topre.values.reshape(1,5))
    topre.rename(columns = {0:"Month",1:"Location",2:"Job_Description",3:"Sector",4:"salary"},inplace=True)
    Z = cv.transform([Eli])
    Z = pd.DataFrame(Z.todense(), columns=cv.get_feature_names())
    topre = pd.concat([topre,Z],axis = 1)
    topre2 = topre
    for i in range(11):
        topre = pd.concat([topre,topre2],axis = 0)
    for i in range(12):
        topre.iloc[i,0] = i+1
    return topre
    
if(__name__) == '__main__':

	data=pd.read_csv("DataSet.csv")
	X=Convert(data)
	to_predict=fun("3000","SALES & MARKETING AGENT","Engineering","DELHI","BTech")

	##for prediction

	##from xgboost import XGBRFRegressor
	model1=xgb.XGBRegressor()

	X_train,X_test,y_train,y_test=train_test_split(X,data["vacancies"],random_state=0)
	model1.fit(X_train,y_train)
	model1.score(X_test,y_test)
