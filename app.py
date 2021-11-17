from flask import Flask, url_for, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_required, logout_user
from flask_login import current_user, login_user
import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField
from wtforms.validators import InputRequired, Length, ValidationError, regexp
from flask_bcrypt import Bcrypt
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from lib.pictures_management import image_to_base64
from lib.user_management import get_current_user_name, get_current_user_avatar
from werkzeug.datastructures import MultiDict
from flask_wtf.csrf import CSRFProtect


# Flask app
app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_data.db'
app.config['SECRET_KEY'] = 'projektstronyinternetowe'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# Login manager
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
deafult_image_path = os.path.join('static', 'imgs', 'default_avatar.png')
deafult_image_base64 = image_to_base64(Image.open(deafult_image_path))

# VAE decoder
VAE_decoder = tf.keras.models.load_model('decoder.h5')

#CSRF Protection
csrf = CSRFProtect(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Database columns
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.now)
    avatar_base64 = db.Column(db.String(20000), nullable=True)

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


class SettingsForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Username"})

    avatar = FileField(render_kw={"placeholder": "Avatar"})

    # [regexp('/([^/]+\.(?:jpg|jpeg|png))')]

    submit = SubmitField("Save changes")
    """
    def validate_username_change(self, username):
        if username == get_current_user_name(User):
            raise ValidationError("Podano tą samą nazwę użytkownika")
        else:
            existing_user_username = User.query.filter_by(
                username=username.data).first()
            if existing_user_username:
                raise ValidationError("Taki użytkownik już istnieje. " +
                                      "Podaj inną nazwę.")
    """

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

    return render_template('dashboard.html', name=get_current_user_name(User),
                           avatar=get_current_user_avatar(User,
                                                          deafult_image_base64))


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/mnistVAE', methods=['GET'])
def mnist_vae():
    x = request.args.get('x')
    y = request.args.get('y')
    if x and y:
        data = np.array([[16 * (float(x) / 1024) - 7,
                          -9.8 * (float(y) / 663) + 4.4]])
        img_data = VAE_decoder.predict(data)
        img_data = img_data[0, :, :, 0]
        img_data = Image.fromarray(np.uint8(img_data * 255), 'L')
        img_str = image_to_base64(img_data)
        return render_template('mnist_vae.html', vae_img=img_str)

    return render_template('mnist_vae.html', vae_img='None')


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():

    form = SettingsForm(
        formdata=MultiDict({'username': get_current_user_name(User)}))

    if form.validate_on_submit():
        image = Image.open(request.files['avatar'].stream)
        image_str = image_to_base64(image, resize_size=(80, 80))
        # todo
        return render_template('test_template.html', image=image_str)
    else:
        print(form.errors)

    return render_template('settings.html', form=form)


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def test():
    x = request.args.get('x')
    return render_template('test_template.html', image=x)


if __name__ == 'main':
    app.run(debug=True)
