from flask import Flask, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import datetime
from markupsafe import escape
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_data.db'
db = SQLAlchemy(app)

#login_manager = LoginManager()
#login_manager.init_app(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.now)

    projects = db.relationship('Project', backref='owner')

    def __repr__(self):
        return f'<User {self.username}>'


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    owner_id = db.Column(db.ForeignKey('user.id'))

    def __repr__(self):
        return f'<Project {self.name}>'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')
