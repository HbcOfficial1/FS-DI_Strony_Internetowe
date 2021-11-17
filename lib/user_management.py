from flask_login import current_user


def get_current_user_name(user_obj):

    cur_username = user_obj.query.filter_by(id=current_user.get_id())
    cur_username = cur_username.first().username

    return cur_username
