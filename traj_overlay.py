import os
from os.path import join

import cv2 as cv
import numpy as np
from PIL import Image

FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS

def sort_files(img_dir):
    sorted_files = sorted(
        os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0])
    )
    return sorted_files

def add_alpha_channel(img_dir, fn):
    img = cv.imread(os.path.join(img_dir, fn), 1)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
    cv.imwrite(os.path.join(img_dir, fn), img)

def convert_to_transparent(img_dir):
    for i, fn in enumerate(sort_files(img_dir)):
        if i == 0 or i == (len(os.listdir(img_dir)) - 1):
            add_alpha_channel(img_dir=img_dir, fn=fn)
            continue
        print(os.path.join(img_dir, fn))
        img = cv.imread(os.path.join(img_dir, fn), 1)
        black_mask = np.all(img == 0, axis=-1)
        alpha = np.uint8(np.logical_not(black_mask)) * 255
        img = np.dstack((img, alpha))
        cv.imwrite(os.path.join(img_dir, fn), img)

        # img = cv.imread(os.path.join(img_dir, fn))
        # img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        # # Slice of alpha channel
        # alpha = img[:, :, 3]
        # alpha[np.all(img[:, :, 0:3] == (0, 0, 0), 2)] = 0
        # cv.imwrite(os.path.join(img_dir, fn), img)

def convert_transparent(img):
    img_f = img
    img = Image.open(img)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save("{}.png".format(img_f.split(".")[0]))


def overlay_pred(img_dir, alpha=0.3, traj_file=None):

    table = Image.open(join(img_dir,"T.png"))
    
    ########## only first one and once; comment out otherwise
    # table = table.convert('RGBA')
    # datas = table.getdata()

    # newData = []
    # for item in datas:
    #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
    #         newData.append((0, 0, 0, 0))
    #     else:
    #         newData.append(item)

    # table.putdata(newData)
    # table.save(os.path.join(img_dir, "T.png"))
    #############

    # convert_transparent(traj_file)
    traj = Image.open(traj_file).convert('RGBA')
    add_alpha_channel(img_dir="", fn=traj_file)

    x, y = table.size
    xp, yp = traj.size
    print(x, y, xp, yp)
    table.paste(traj, (x, y))
    result = Image.blend(table, traj, alpha=alpha)
    result.save(os.path.join(img_dir, "T.png"))

if __name__ == "__main__":
    traj_file = "/home/eleyng/Videos/bclstmgmm/up/5.png"
    img_dir = "/home/eleyng/Videos/bclstmgmm/up/"  # Input image dir, created by Blender
    alpha = 0.3  # Transparency (between 0-1; 0 less and 1 is more opacity)
    skip_frames = 0  # Overlay every skip_frames
    overlay_pred(img_dir, alpha=alpha, traj_file=traj_file)
