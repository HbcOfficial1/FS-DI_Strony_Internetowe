import base64
from io import BytesIO
import PIL
from typing import Optional, Tuple


def image_to_base64(PIL_image: PIL.Image, resize_size: Optional[Tuple] = None,
                    image_format: Optional[str] = 'png'):

    image = PIL_image

    buffered = BytesIO()

    if resize_size:
        image = image.resize(resize_size)

    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(img_str)[2:-1]

    return img_str



