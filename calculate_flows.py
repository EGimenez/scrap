import cv2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage import color
from skimage import io
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
import pickle
import portion as I

filename_in = '20211215_145526_EA4F.mp4'
filename_out = '20211215_145526_EA4F_out.mp4'
filename_out_2 = '20211215_145526_EA4F_out_2.mp4'
filename_out_3 = '20211215_145526_EA4F_out_3.mp4'

fps = 30
the_intervals = I.closed(fps*30, fps*60) | I.closed(fps*120, fps*130) | I.closed(fps*160, I.inf)


def flow_calculator_cache(image0_c, image1_c, i):
    image0 = color.rgb2gray(image0_c)
    image1 = color.rgb2gray(image1_c)

    image0 = resize(image0, (image0.shape[0] // 2, image0.shape[1] // 2), anti_aliasing=True)
    image1 = resize(image1, (image1.shape[0] // 2, image1.shape[1] // 2), anti_aliasing=True)

    v, u = optical_flow_ilk(image0, image1, radius=15)

    with open('vector_field_15_rscale_2/vector_field_{}.pickle'.format(str(i).zfill(6)), 'wb') as handle:
        pickle.dump({'v': v, 'u': u}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def flow_calculator(image0_c, image1_c):
    image0 = color.rgb2gray(image0_c)
    image1 = color.rgb2gray(image1_c)

    image0 = resize(image0, (image0.shape[0] // 6, image0.shape[1] // 6), anti_aliasing=True)
    image1 = resize(image1, (image1.shape[0] // 6, image1.shape[1] // 6), anti_aliasing=True)

    # --- Compute the optical flow
    v, u = optical_flow_ilk(image0, image1, radius=15)
    # v, u = optical_flow_tvl1(image0, image1)

    # --- Compute flow magnitude
    norm = np.sqrt(u ** 2 + v ** 2)

    # --- Quiver plot arguments
    nvec = 40  # Number of vectors to be displayed along each image dimension
    nl, nc = image0.shape
    step = max(nl // nvec, nc // nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    u_ /= 5
    v_ /= 5

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.imshow(norm)
    ax.quiver(x, y, u_, v_, color='r', units='dots',
              angles='xy', scale_units='xy', lw=3)

    plt.savefig('pepito.png')
    image = mpimg.imread('pepito.png')
    image = image[:, :, :3]

    plt.close(fig)
    plt.cla()

    image0_c_r = resize(image0_c, (200, 400), anti_aliasing=True)

    image = np.concatenate((image0_c_r, image), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.imshow(image)
    plt.savefig('pepito_concat.png')
    plt.close(fig)

    image = 255 * image  # Now scale by 255
    image = image.astype(np.uint8)

    image0_c_r = 255 * image0_c_r  # Now scale by 255
    image0_c_r = image0_c_r.astype(np.uint8)


    return image, image0_c_r


def process_file_cache():
    video_in = cv2.VideoCapture(filename_in)

    ret, frame_0 = video_in.read()

    i = 0
    while video_in.isOpened():
        try:
            ret, frame_1 = video_in.read()
        except:
            break

        if ret:
            if i in the_intervals:
                flow_calculator_cache(frame_0, frame_1, i)
                frame_0 = frame_1
            i += 1
            print(i)
        else:
            break

    video_in.release()


def process_file():
    video_in = cv2.VideoCapture(filename_in)

    frame_width = int(video_in.get(3))
    frame_height = int(video_in.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = int(video_in.get(cv2.CAP_PROP_FPS))

    video_out = cv2.VideoWriter(filename_out,
                                fourcc,
                                fps,
                                (800, 200))

    video_out_2 = cv2.VideoWriter(filename_out_2,
                                fourcc,
                                fps,
                                (frame_width, frame_height))
    video_out_3 = cv2.VideoWriter(filename_out_3,
                                fourcc,
                                fps,
                                (400, 200))

    if not video_in.isOpened():
        print("Error opening video stream or file")


    ret, frame_0 = video_in.read()

    i = 0
    while video_in.isOpened():
        try:
            ret, frame_1 = video_in.read()
            # ret, frame_1 = video_in.read()
            # ret, frame_1 = video_in.read()
        except:
            break

        if ret:
            frame, frame_1_2 = flow_calculator(frame_0, frame_1)
            video_out.write(frame)
            video_out_2.write(frame_1)
            video_out_3.write(frame_1_2)
            frame_0 = frame_1
            i += 1
            print(i)
            # if i >= 1000:
            #     break
        else:
            break

    video_in.release()
    video_out.release()
    video_out_2.release()
    video_out_3.release()


if __name__ == '__main__':
    # process_file()
    process_file_cache()
