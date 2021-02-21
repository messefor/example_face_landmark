'''Face recognition


  OpenCV
  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
'''

import time
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import dlib

from collections import defaultdict

DEVICE_ID = 0

# fname = 'data/aa.csv'
# values = [[1, 2, 3, 4], [0, 4, 5, 4]]
# np.savetxt(fname, np.array(values), delimiter=',', newline='\t')


size = (640, 360)
cap = cv2.VideoCapture(DEVICE_ID)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f'size=({width}, {height}) FPS={fps}')


predictor_fpath = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_fpath)

window_size = 30

pooling_started = False
while(True):

  # Capture frame-by-frame
  ret, frame = cap.read()

  if not pooling_started:
    start_ts = datetime.datetime.now().timestamp()
    pooling_started = True
    values = []

  # Our operations on the frame come here
  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # frame = cv2.resize(frame, size)

  detected = detector(frame[:, :, ::-1])

  if len(detected) > 0:

    # Face landmark
    parts = predictor(frame, detected[0]).parts()
    for p in parts:
        cv2.circle(frame, (p.x, p.y), 1, (255, 0, 0), -1)

    op_ro = parts[41].y - parts[37].y  # Right eye outsize
    op_ri = parts[40].y - parts[38].y  # Right eye inside
    op_li = parts[47].y - parts[43].y  # Left eye inside
    op_lo = parts[46].y - parts[44].y  # Left eye outside
    v = [op_ro, op_ri, op_li, op_lo]

    values.append(v)

  # Display the resulting frame
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  end_dt = datetime.datetime.now()
  if end_dt.timestamp() - start_ts > window_size:
    dtstr = end_dt.strftime('%H%M%S')
    fname = f'data/openinig_{dtstr}.csv'
    values = np.array(values)
    np.savetxt(fname, values, fmt='%.1f', delimiter=',', newline='\n')
    print(f'Saved: {fname}')
    pooling_started = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------

# fname = 'data/openinig_183645.csv'
# d = pd.read_csv(fname, header=None)
# ax = d.plot()
# ax.get_figure().savefig(fname.split('.')[0] + '.png')
