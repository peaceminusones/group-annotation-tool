# from math import exp
# import numpy as np
# import cv2
# import os



from pyheatmap.heatmap import HeatMap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# file_path = ['./csv/out_013_1_1.mp4.csv', 
#              './csv/out_013_2_1.mp4.csv']

# file_path = ['./csv/out_0923_1.mp4.csv', 
#              './csv/out_0923_2.mp4.csv',
#              './csv/out_0923_3.mp4.csv']

# file_path = ['./csv/out_1939_1.mp4.csv',
#              './csv/out_1939_2.mp4.csv']

# data = []
# for path in file_path:
#     img_message = pd.read_csv(path)
#     all_track_id = img_message.loc[:,"track_id"].values
#     all_track_id = list(set(all_track_id))
#     for i in all_track_id:
#         x = img_message[(img_message.track_id == int(i))].x.values
#         y = img_message[(img_message.track_id == int(i))].y.values
#         for l in range(len(x)):
#             tmp = [int(x[l]/2), int(y[l]/2), 100]
#             data.append(tmp)

# # N = 10000
# # X = np.random.rand(N) * 255  # [0, 255]
# # Y = np.random.rand(N) * 255
# # data = []
# # for i in range(N):
# #     tmp = [int(X[i]), int(Y[i]), 1]
# #     data.append(tmp)

# heat = HeatMap(data)

# heat.heatmap(save_as="./images/heatmap3.png") #热图

# data = []
# N = 1000
# X = np.random.rand(N) * 1280  # [0, 255]
# Y = np.random.rand(N) * 720
# data = []
# for i in range(N):
#     tmp = [int(X[i]), int(Y[i]), 0]
#     data.append(tmp)
#     for m in range(0, 20, 1):
#         for n in range(330, 350, 1):
#             tmp = [m, n, 1]
#             data.append(tmp)


# heat = HeatMap(data)

# heat.heatmap(save_as="./images/heatmap4.png") #热图




import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
from scipy.stats import multivariate_normal

file_path = ['./csv/out_013_1_1.mp4.csv', 
             './csv/out_013_2_1.mp4.csv']

# file_path = ['./csv/out_0923_1.mp4.csv', 
#              './csv/out_0923_2.mp4.csv',
#              './csv/out_0923_3.mp4.csv']

# file_path = ['./csv/out_1939_1.mp4.csv',
#              './csv/out_1939_2.mp4.csv']

def gauss_fun(X, Y):
    # 大厅
    # mux = [0, 20, 20, 750, 800, 1250, 1280]
    # muy = [340, 300, 480, 0, 60, 400, 550]
    # sx = [60, 100, 80, 60, 80, 50, 80]
    # sy = [60, 100, 80, 60, 80, 60, 70]
    # rho = [0, 0.3, 0.2, 0.1, 0, 0.2, 0.3]

    mux = []
    muy = []
    sx = []
    sy = []
    rho = []

    for path in file_path:
        img_message = pd.read_csv(path)
        all_track_id = img_message.loc[:,"track_id"].values
        all_track_id = list(set(all_track_id))
        for i in all_track_id:
            x = img_message[(img_message.track_id == int(i))].x.values
            y = img_message[(img_message.track_id == int(i))].y.values

            mux.append(int(x[0]))
            muy.append(int(y[0])+50)
            sx.append(50)
            sy.append(50)
            rho.append(0)
            mux.append(int(x[-1]))
            muy.append(int(y[-1]+100))
            sx.append(50)
            sy.append(40)
            rho.append(0.1)
    print(muy)
    # 广场
    # mux = [0,   0,   180, 320, 630, 780, 1200]
    # muy = [300, 200, 720, 700, 180, 200, 720]
    # sx =  [80,  70,  90,  75,  75,  80,  60]
    # sy =  [65,  60,  80,  60,  60,  60,  60]
    # rho = [0.2, 0,   0.6, 0.1, 0,   0.2, 0.1]

    # 食堂
    # mux = [80,  0,   550, 650,  800, 900,  0,    200, 900, 1000, 1200, 1280, 1280]
    # muy = [350, 300, 120, 130,  150, 100,  720,  720, 720, 720,  700,  350,  450]
    # sx =  [60,  90,  85,  60,   85,  65,   100,  70,  70,  90,   80,   60,   80]
    # sy =  [60,  80,  80,  55,   70,  58,   90,   70,  55,  90,   50,   75,   80]
    # rho = [0,   0.1, 0,   0.12, 0.2, 0.05, 0.18, 0,   0.1, 0,    0.2,  0,    0.3]

    d = np.dstack([X, Y])
    z = None
    for i in range(len(mux)):
        mean = [mux[i], muy[i]]
        # Extract covariance matrix
        cov = [[sx[i] * sx[i], rho[i] * sx[i] * sy[i]], [rho[i] * sx[i] * sy[i], sy[i] * sy[i]]]
        gaussian = multivariate_normal(mean = mean, cov = cov)
        z_ret = gaussian.pdf(d)

        if z is None:
            z = z_ret
        else:
            z += z_ret

    return z

x = np.linspace(0, 1280, 100)
y = np.linspace(0, 720, 100)

X, Y = np.meshgrid(x, y)
Z = gauss_fun(X, Y)


fig, ax = plt.subplots(figsize=(10, 72/128*10))
# plt.contour(X, Y, Z, 10, colors='grey')
plt.contourf(X, Y, Z, 20, cmap='RdBu')
plt.colorbar()
plt.xlim(0,1280)
plt.ylim(0,720)
ax.invert_yaxis()
plt.show()