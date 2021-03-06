from flask import Flask, url_for, render_template, redirect, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_required, logout_user
from flask_login import current_user, login_user
import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField
from wtforms import FloatField, IntegerRangeField, IntegerField
from wtforms.validators import InputRequired, Length, ValidationError
from wtforms.validators import NumberRange
from flask_bcrypt import Bcrypt
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from lib.pictures_management import image_to_base64
from lib.user_management import get_current_user_name, get_current_user_avatar
from lib.DeepDreamModel import run_deepdream
from flask_wtf.csrf import CSRFProtect

# from dotenv import load_dotenv
# dotenv_path = '.env'  # Path to .env file
# load_dotenv(dotenv_path)

# Define port
port = int(os.environ.get('PORT', 5000))

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
VAE_decoder = tf.keras.models.load_model(os.path.join('models', 'decoder.h5'))

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
    avatar_base64 = db.Column(db.Text(), nullable=True)

    projects = db.relationship('Project', backref='owner')

    def __repr__(self):
        return f'<User {self.username}>'


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    input_img = db.Column(db.Text(), nullable=True)
    output_img = db.Column(db.Text(), nullable=True)
    octave = db.Column(db.Float(precision=2), nullable=True)
    octaves_minus = db.Column(db.Integer(), nullable=True)
    octaves_plus = db.Column(db.Integer(), nullable=True)
    steps = db.Column(db.Integer(), nullable=True)

    owner_id = db.Column(db.ForeignKey('user.id'))


    def __repr__(self):
        return f'<Project {self.name}>'


# Forms
class RegisterForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Nazwa u??ytkownika"})

    password = PasswordField(validators={InputRequired(),
                             Length(min=4, max=20)},
                             render_kw={"placeholder": "Has??o"})

    submit = SubmitField("Zarejestruj")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError("Taki u??ytkownik ju?? istnieje. Podaj inn?? " +
                                  "nazw??.")


class LoginForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Nazwa u??ytkownika"})

    password = PasswordField(validators={InputRequired(),
                             Length(min=4, max=20)},
                             render_kw={"placeholder": "Has??o"})

    submit = SubmitField("Zaloguj")


class SettingsForm(FlaskForm):
    username = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Nazwa u??ytkownika"})

    # todo add regex validator for image files
    avatar = FileField(render_kw={"placeholder": "Avatar"})

    cur_password = PasswordField(validators={Length(max=20)},
                             render_kw={"placeholder": "Has??o"})

    new_password = PasswordField(validators={Length(max=20)},
                             render_kw={"placeholder": "Nowe has??o"})

    new_password_again = PasswordField(validators={Length(max=20)},
                             render_kw={"placeholder": "Powt??rz has??o"})


    submit = SubmitField("Zapisz zmiany")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username and existing_user_username.username != \
                get_current_user_name(User):
            raise ValidationError("Taki u??ytkownik ju?? istnieje. Podaj inn?? " +
                                  "nazw??.")

    def validate_new_password(self, new_password):
        if new_password.data != '' and new_password.data is not None:
            if len(new_password.data) < 4:
                raise ValidationError("Nowe has??o powinno mie?? ponad 4 znaki.")
            if new_password.data != self.new_password_again.data:
                raise ValidationError("Podane has??a nie s?? zgodne z sob??.")

    def validate_cur_password(self, cur_password):
        if cur_password.data != '' and cur_password.data is not None:
            user = User.query.filter_by(
                username=get_current_user_name(User)).first()
            if not bcrypt.check_password_hash(user.password, cur_password.data):
                raise ValidationError("Podano b????dne has??o do konta.")


class ProjectForm(FlaskForm):

    # todo add regex validator for image files
    input_img = FileField(render_kw={"placeholder": "input_img"})

    name = StringField(validators={InputRequired(), Length(min=4, max=20)},
                           render_kw={"placeholder": "Nazwa projektu"})

    octave = FloatField(validators={InputRequired(),
                                    NumberRange(min=1.2, max=3)})

    octaves_minus = IntegerRangeField(validators={InputRequired(),
                                            NumberRange(min=-3, max=0)})

    octaves_plus = IntegerRangeField(validators={InputRequired(),
                                            NumberRange(min=0, max=3)})


    steps = IntegerField(validators={InputRequired(),
                                     NumberRange(min=1, max=200)})

    submit = SubmitField("Dodaj projekt")


class DeleteProjectForm(FlaskForm):
    submit = SubmitField("Usu?? projekt")


# Routes and redirects
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
        new_user = User(username=form.username.data,
                        password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Konto zosta??o utworzone', category='accept')
        return redirect(url_for('login'))

    else:
        for error in form.errors.values():
            flash(error[0], category='deny')

    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash('Pomy??lnie zalogowano!', category='accept')
                return redirect(url_for('dashboard'))
        flash('Nieprawid??owa nazwa u??ytkownika lub has??o.', category='deny')

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

    form = SettingsForm(username=get_current_user_name(User))

    if form.validate_on_submit():

        changes = False

        user = User.query.filter_by(username=get_current_user_name(
            User)).first()

        if request.files['avatar']:
            image = Image.open(request.files['avatar'].stream)
            image_str = image_to_base64(image, resize_size=(80, 80))
            user.avatar_base64 = image_str
            changes = True

        if form.username.data != get_current_user_name(User):
            user.username = form.username.data
            changes = True

        if form.new_password.data and form.new_password_again.data:
            user.password = bcrypt.generate_password_hash(form.new_password.data)
            changes = True

        if changes:
            db.session.commit()
            flash('Zmiany zosta??y zaakceptowane.', category='accept')
            return redirect(url_for('dashboard'))

    else:
        for error in form.errors.values():
            flash(error[0], category='deny')

    return render_template('settings.html', form=form)


@app.route('/projects', methods=['GET', 'POST'])
@login_required
def projects():

    user_projects = Project.query.filter_by(owner_id=current_user.get_id())

    return render_template('projects.html', user_projects=user_projects)

@app.route('/newproject', methods=['GET', 'POST'])
@login_required
def newproject():

    form = ProjectForm(octave=1.3, steps=100, octaves_minus=-1, octaves_plus=1)

    if form.validate_on_submit():

        user_id = current_user.get_id()

        img_input = Image.open(request.files['input_img'].stream)
        project_name = form.name.data
        ocataves_range = range(form.octaves_minus.data,
                               form.octaves_plus.data + 1)
        octave_scale = form.octave.data
        epochs_per_octave = form.steps.data

        image = run_deepdream(img_input, octaves=ocataves_range,
                              octave_scale=octave_scale,
                              steps_per_octave=epochs_per_octave)

        image_output_base64 = image_to_base64(image)
        image_input_base64 = image_to_base64(img_input)

        project = Project(name=project_name, input_img=image_input_base64,
                          output_img=image_output_base64, octave=octave_scale,
                          octaves_minus=form.octaves_minus.data,
                          octaves_plus=form.octaves_plus.data,
                          steps=epochs_per_octave, owner_id=user_id)

        db.session.add(project)
        db.session.commit()
        flash('Utworzono nowy projekt', category='accept')
        return redirect(url_for('projects'))

    return render_template('newproject.html', form=form)


@app.route('/project', methods=['GET', 'POST'])
@login_required
def project():

    project_id = request.args.get('id')

    project = Project.query.filter_by(id=project_id).first()

    if project is not None:
        if str(project.owner_id) == current_user.get_id():

            delete_button_form = DeleteProjectForm()

            if delete_button_form.validate_on_submit():
                db.session.delete(project)
                db.session.commit()
                flash("Pomy??lnie usuni??to projekt.", category='accept')
                return redirect(url_for('projects'))

            return render_template('view_project.html', project=project,
                                   form=delete_button_form)
        else:
            flash('Nie masz dost??pu do tego projektu.', category='deny')
            return redirect(url_for('projects'))
    else:
        flash('Projekt nie istnieje.', category='deny')
        return redirect(url_for('projects'))


if __name__ == 'main':
    app.run(debug=True, host='0.0.0.0', port=port)
