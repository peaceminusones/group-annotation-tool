
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = ['./csv/out_013_1_1.mp4.csv', 
             './csv/out_013_2_1.mp4.csv', 
             './csv/out_0923_1.mp4.csv', 
             './csv/out_0923_2.mp4.csv',
             './csv/out_0923_3.mp4.csv',
             './csv/out_1939_1.mp4.csv',
             './csv/out_1939_2.mp4.csv']
file_path2 = ['./csv/out_1939_1.mp4.csv',
              './csv/out_1939_2.mp4.csv']

# dic = {}
length = 0
result = []
for path in file_path:
    img_message = pd.read_csv(path)
    all_track_id = img_message.loc[:,"track_id"].values
    new_track_id = list(set(all_track_id))
    for i in new_track_id:
        choosebytrackid = img_message[(img_message.track_id == int(i))]
        if path in file_path2:
            if 3*len(choosebytrackid) >= 18:
                length+=2
            result.append(3*len(choosebytrackid))
            result.append(3*len(choosebytrackid))
        else:
            if len(choosebytrackid) >= 18:
                length+=2
            result.append(len(choosebytrackid))
            result.append(len(choosebytrackid))
        # length = len(choosebytrackid)
        # if length in dic.keys():
        #     dic[length] += 2
        # else:
        #     dic[length] = 2

# vals, freqs = dic.Render()
# print(vals, freqs)
print(length/len(result))
sns.distplot(result, bins = 20, kde = False, hist_kws = {'color':'purple'})
plt.xlabel('trajectory length')
plt.ylabel('number')
plt.show()



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math

# file_path = [
#              './csv/out_013_1_1.mp4.csv', 
#              './csv/out_013_2_1.mp4.csv', 
#              './csv/out_0923_1.mp4.csv', 
#              './csv/out_0923_2.mp4.csv',
#              './csv/out_0923_3.mp4.csv',
#              './csv/out_1939_1.mp4.csv',
#              './csv/out_1939_2.mp4.csv',
#              ]

# # dic = {}
# result = []
# for path in file_path:
#     img_message = pd.read_csv(path)
#     all_track_id = img_message.loc[:,"track_id"].values
#     new_track_id = list(set(all_track_id))
#     for i in new_track_id:
#         x = img_message[(img_message.track_id == int(i))].x.values
#         y = img_message[(img_message.track_id == int(i))].y.values
#         vx = abs(x[-1] - x[0]) / len(x) / 40
#         vy = abs(y[-1] - y[0]) / len(y) / 40
#         v = math.sqrt(vx**2 + vy**2)
#         result.append(v)
#         if len(x) <= 2:
#             continue
        
#         vx = abs(x[-2] - x[0]) / (len(x)-1) / 34
#         vy = abs(y[-2] - y[0]) / (len(y)-1) / 34
#         v = math.sqrt(vx**2 + vy**2)
#         result.append(v)
#         # length = len(choosebytrackid)
#         # if length in dic.keys():
#         #     dic[length] += 2
#         # else:
#         #     dic[length] = 2

# # vals, freqs = dic.Render()
# # print(vals, freqs)
# print(sum(result)/len(result))
# sns.distplot(result, bins = 20, kde = False, hist_kws = {'color':'purple'})
# plt.xlabel('velocity (m/s)')
# plt.ylabel('number')
# plt.show()