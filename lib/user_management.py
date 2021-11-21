from flask_login import current_user
from pandas import isnull

def get_current_user_name(user_obj):

    cur_username = user_obj.query.filter_by(id=current_user.get_id())
    cur_username = cur_username.first().username

    return cur_username


def get_current_user_avatar(user_obj, deafult_image_base64):

    cur_username = user_obj.query.filter_by(id=current_user.get_id())
    avatar = cur_username.first().avatar_base64

    if isnull(avatar):
        return deafult_image_base64
    else:
        return avatar
