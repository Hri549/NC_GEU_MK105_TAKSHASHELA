from flask import Flask,render_template,request, session, redirect, url_for,flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os,time
import matplotlib.pyplot as plt

import Sih_try as func

import pandas as pd
import xgboost as xgb

data = pd.read_csv("DataSet.csv")

from flask_wtf import FlaskForm,RecaptchaField
'''from wtforms import (StringField,SubmitField,
                     DateTimeField, RadioField,
                     SelectField,TextAreaField, DateField)'''

from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/sih_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

app.config["SECRET_KEY"] = "mysecretkey"

app.config["RECAPTCHA_PUBLIC_KEY"] = "6Lesd7gZAAAAAKVU6WXhPxdbWatc4nKvaWY8Or5G" #"KEY GOES HERE "
app.config["RECAPTCHA_PRIVATE_KEY"] = "6Lesd7gZAAAAAMh2Tk7l_5oroVbh2kHysGXmG_tS" #"PRIVATE KEY GOES HEREE"

engine = create_engine("mysql://root:@localhost/sih_data")

class User(db.Model):
	sr_no = db.Column(db.Integer, primary_key = True)
	
	name = db.Column(db.String(80), unique = False, nullable=False)
	email = db.Column(db.String(20),unique = True, nullable=False)
	password = db.Column(db.String(20),unique = False, nullable=False)
	city = db.Column(db.String(20),unique = False, nullable=False)

class Widgets(FlaskForm):
	recaptcha = RecaptchaField()


@app.route("/",methods = ['GET','POST'])
def index():
	
	return render_template('index.html')

@app.route("/signup",methods = ['GET','POST'])
def signup():
	form = Widgets()
	if(request.method == "POST"):
		Vname = request.form.get('name')
		Vemail = request.form.get('e-mail')
		Vpassword = request.form.get('password')
		Vcity = request.form.get('city')
		entry = User( name = Vname ,email = Vemail, password = Vpassword, city = Vcity)
		db.session.add(entry)
		db.session.commit()
		return redirect(url_for('home'))
	return render_template('signup.html')

@app.route("/login",methods = ['GET','POST'])
def login():
	if(request.method == "POST"):
		Login_email = str(request.form.get('e-mail'))
		Login_password = str(request.form.get('password'))
		Session = sessionmaker(bind=engine)
		s = Session()
		query = s.query(User).filter(User.email.in_([Login_email]), User.password.in_([Login_password]))
		result = query.first()
		if(result):
			return redirect(url_for('home'))
		else:
			flash(u"Wrong email or password!!!","warning")
	return render_template('login.html')
	
@app.route("/home",methods = ['GET','POST'])
def home():
	if(request.method == 'POST'):
		Uedu = str(request.form.get('education'))
		Ujob = str(request.form.get('job-title'))
		Usec = str(request.form.get('sector'))
		Ucty = str(request.form.get('city'))

		Usal = request.form.get('Salary')
		
		X = func.Convert(data)
		
		y = data["vacancies"]
		X_pred = func.fun(Usal,Ujob,Usec,Ucty,Uedu)
		model=xgb.XGBRegressor()
		model.fit(X,y)
		pre = model.predict(X_pred)
		plt.plot(pre)
		global new_graph_name
		new_graph_name = "graph" + str(time.time()) + ".png"
	

		for filename in os.listdir('static/'):
			if filename.startswith('graph'):
				os.remove('static/' + filename)
	
		plt.savefig('static/' + new_graph_name)

		return redirect(url_for('returna'))
	
	return render_template('home.html')
	
@app.route("/result",methods = ['GET','POST'])
def returna():

	'''data = [9,7,2,8,1]
	plt.plot(data)
	new_graph_name = "graph" + str(time.time()) + ".png"
	

	for filename in os.listdir('static/'):
		if filename.startswith('graph_'):
			os.remove('static/' + filename)
	
	plt.savefig('static/' + new_graph_name)'''
	global new_graph_name
	
	return render_template('result.html',graph = new_graph_name)




@app.route("/trending",methods = ['GET','POST'])
def trending():
	
	return render_template('trending.html')

app.run(debug = True)

