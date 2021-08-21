import cv2
from turbojpeg import (TurboJPEG, TJPF_GRAY, \
        TJSAMP_GRAY, TJFLAG_PROGRESSIVE)


def img_dec_TurboJPEG(img_path):

    # specifying library path explicitly
    # jpeg = TurboJPEG(r'D:\turbojpeg.dll')
    # jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')
    # jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')

    # using default library installation
    jpeg = TurboJPEG()

    # decoding input.jpg to BGR array
    with open(img_path, 'rb') as in_file:
        bgr_array = jpeg.decode(in_file.read())
    return bgr_array
    # print('bgr_array', type(bgr_array), bgr_array.shape)


# ======================================================= #
if __name__ == '__main__':
    img_path = ""
    img_dec_TurboJPEG(img_path)