from flask import Flask, url_for, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_required, logout_user
from flask_login import current_user, login_user
import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

# Flask app
app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_data.db'
app.config['SECRET_KEY'] = 'projektstronyinternetowe'
db = SQLAlchemy(app)

# Login manager
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Database columns
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
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

# Forms
class RegisterForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Username"})

    password = PasswordField(validators={InputRequired(),
                             Length(min=4, max=20)},
                             render_kw={"placeholder": "Password"})

    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError("Taki użytkownik już istnieje. Podaj inną " +
                                  "nazwę.")


class LoginForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Username"})

    password = PasswordField(validators={InputRequired(),
                             Length(min=4, max=20)},
                             render_kw={"placeholder": "Password"})

    submit = SubmitField("Login")


# Routes and redicrects
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/mnistVAE')
@login_required
def mnist_vae():
    return render_template('mnist_vae.html')


if __name__ == 'main':
    app.run(debug=True)
