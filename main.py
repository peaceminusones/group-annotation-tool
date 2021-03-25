"""
Copyright {2018} {Viraj Mavani}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
"""

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk

import os
import numpy as np
import tensorflow as tf
import config
import math
import pandas as pd
import cv2
import filetype

# # distort matrix ----------------------------------------------
# mtx = np.array([[981.0451,0.,686.7909],
# 		        [0.,993.9882,342.1298],
# 		        [0.,0.,1.]])
# dist = np.array([[-0.4216,0.2277,-0.00000,-0.00000,0.00000]])
# # -------------------------------------------------------------

class MainGUI:
    def __init__(self, master):
        self.parent = master
        self.parent.title("Group Annotation Tool")
        self.frame = Frame(self.parent)
        # expand = 1:打开fill属性
        # fill=X 当GUI窗体大小发生变化时，widget在X方向跟随GUI窗体变化 
        # fill=Y 当GUI窗体大小发生变化时，widget在Y方向跟随GUI窗体变化 
        # fill=BOTH 当GUI窗体大小发生变化时，widget在X、Y两方向跟随GUI窗体变化
        self.frame.pack(fill=BOTH, expand=1) 
        # self.parent.resizable(width=False, height=False)
        self.rate = 0
        # Initialize class variables
        self.img_message = pd.DataFrame()
        # self.img_message['group_id'] = None
        # self.max_track_id = self.img_message.loc[:,"track_id"].max()
        self.frameId = 0
        self.img = None
        self.tkimg = None
        self.imgfile = ''
        self.videoDir= ''
        self.imageDir = ''
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.bboxList = []
        self.bboxIdgroupList = []
        self.bboxgroupList = []
        self.bboxPointList = []
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.bboxId = None
        self.boxtrackid = dict()
        self.bboxIdgroup = None
        self.boxgroupid = dict()
        self.currLabel = None
        self.editbboxId = None
        self.currBboxColor = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None
        self.vl = None
        self.editPointId = None
        self.filename = None
        self.filenameBuffer = None
        self.objectLabelList = []
        self.EDIT = False

        self.globalfiledir = ''

        # initialize mouse state
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        # initialize annotation file
        self.anno_filename = 'annotations.csv'
        self.annotation_file = open('annotations/' + self.anno_filename, 'w+')
        self.annotation_file.write("")
        self.annotation_file.close()

        # ------------------ GUI ---------------------

        # Control Panel
        self.ctrlPanel = Frame(self.frame)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        
        self.openViBtn = Button(self.ctrlPanel, text='Open video')
        self.openViBtn.pack(fill=X, side=TOP)
        self.openViBtn.bind('<Button-1>',self.create_video_frame)
        
        self.openBtn = Button(self.ctrlPanel, text='Open frame')
        self.openBtn.pack(fill=X, side=TOP)
        self.openBtn.bind('<Button-1>',self.open_image)
        
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir')
        self.openDirBtn.pack(fill=X, side=TOP)
        self.openDirBtn.bind('<Button-1>',self.open_image_dir)
        
        self.nextBtn = Button(self.ctrlPanel, text='Next-->')
        self.nextBtn.pack(fill=X, side=TOP)
        self.nextBtn.bind('<Button-1>',self.open_next)

        self.previousBtn = Button(self.ctrlPanel, text='<--Previous')
        self.previousBtn.pack(fill=X, side=TOP)
        self.previousBtn.bind('<Button-1>',self.open_previous)

        # self.saveBtn = Button(self.ctrlPanel, text='Save')
        # self.saveBtn.pack(fill=X, side=TOP)
        # self.saveBtn.bind('<Button-1>',self.save)

        self.disp = Label(self.ctrlPanel, text='Coordinates:')

        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View")
        self.zoomPanelLabel.pack(fill=X, side=TOP)

        self.zoomcanvas = Canvas(self.ctrlPanel, width=150, height=150) 
        self.zoomcanvas.pack(fill=X, side=TOP, anchor='center')

        self.newtrackidlabel = Label(self.ctrlPanel, text="New track Id")
        self.newtrackidlabel.pack(fill=X, side=TOP)
        self.newtrackid = Entry(self.ctrlPanel)
        self.newtrackid.pack(fill=X, side=TOP,anchor='center')

        # self.newgroupidlabel = Label(self.ctrlPanel, text="New group Id ")
        # self.newgroupidlabel.pack(fill=X, side=TOP)
        # self.newgroupidentry = Entry(self.ctrlPanel)
        # self.newgroupidentry.pack(fill=X, side=TOP,anchor='center')

        # input group id
        self.spacelable1 = Label(self.ctrlPanel, text="           ").pack(fill=X, side=TOP)
        self.spacelable2 = Label(self.ctrlPanel, text="           ").pack(fill=X, side=TOP)
        self.labelTrackId = Label(self.ctrlPanel, text="track id(split by space)").pack(fill=X, side=TOP)
        self.entryTrackId = Entry(self.ctrlPanel)
        self.entryTrackId.pack(fill=X, side=TOP,anchor='center')
        
        self.labelGroupId = Label(self.ctrlPanel, text="Group id")
        self.labelGroupId.pack(fill=X, side=TOP)
        self.entryGroupId = Entry(self.ctrlPanel)
        self.entryGroupId.pack(fill=X, side=TOP,anchor='center')
        self.buttongroup = Button(self.ctrlPanel, text="Group", command=self.button_groupid).pack(fill=X, side=TOP)
        
        # Image Editing Region
        self.canvas = Canvas(self.frame, width=800, height=500)
        self.canvas.grid(row=0, column=1, sticky=W + N) # sticky 用于拉伸对齐
        self.canvas.bind("<Button-1>", self.mouse_click)  # <Button-1>:鼠标左键单击
        self.canvas.bind("<Motion>", self.mouse_move, "+") # <Motion>:鼠标移动
        self.canvas.bind("<B1-Motion>", self.mouse_drag)   # <B1-Motion>:按住鼠标左键移动
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release) # <ButtonRelease-1>:鼠标左键被释放
        self.parent.bind("<Key-Left>", self.open_previous)   # “方向键—>”：下一幅图片
        self.parent.bind("<Key-Right>", self.open_next)      # “方向键<—”：上一幅图片
        self.parent.bind("Escape", self.cancel_bbox)

        # Labels and Bounding Box Lists Panel
        self.listPanel = Frame(self.frame)
        self.listPanel.grid(row=0, column=2, sticky=W + N)
        self.listBoxNameLabel = Label(self.listPanel, text="List of Objects").pack(fill=X, side=TOP)
        self.objectListBox = Listbox(self.listPanel, width=40)
        self.objectListBox.pack(fill=X, side=TOP)
        self.delObjectBtn = Button(self.listPanel, text="Delete", command=self.del_bbox)
        self.delObjectBtn.pack(fill=X, side=TOP)
        self.clearAllBtn = Button(self.listPanel, text="Clear All", command=self.clear_all)
        self.clearAllBtn.pack(fill=X, side=TOP)

        # Labels and bounding box list group
        self.listgroupNameLabel = Label(self.listPanel, text="List of Groups").pack(fill=X, side=TOP)
        self.objectListgroup = Listbox(self.listPanel, width=40)
        self.objectListgroup.pack(fill=X, side=TOP)
        self.delgroupBtn = Button(self.listPanel, text="Delete", command=self.del_bboxgroup)
        self.delgroupBtn.pack(fill=X, side=TOP)
        self.clearAllgroupBtn = Button(self.listPanel, text="Clear All", command=self.clear_allgroup)
        self.clearAllgroupBtn.pack(fill=X, side=TOP)

        # Finish and save to .csv
        self.ctrlFinish = Frame(self.frame)
        self.ctrlFinish.grid(row=1, column=2, sticky=W + N)
        self.finishBtn = Button(self.ctrlFinish, width=39, text="Save All", command=self.save_to_csv)
        self.finishBtn.pack(fill=X, side=TOP)

        # STATUS BAR
        self.statusBar = Frame(self.frame, width=800)
        self.statusBar.grid(row=1, column=1, sticky=W + N)
        self.processingLabel = Label(self.statusBar, text="                      ")
        self.processingLabel.pack(side="left", fill=X)
        self.imageIdxLabel = Label(self.statusBar, text="                      ")
        self.imageIdxLabel.pack(side="right", fill=X)

    """
    读取视频，存储成帧，存储位置在"./frame data"
    """
    def video_to_frame(self, filepath):
        base_name = os.path.basename(filepath) # 读取当前文件名
        parent_dir = os.path.dirname(os.path.dirname(filepath))  # 读取父级目录
        video = cv2.VideoCapture(filepath)
        c = 0
        rval = video.isOpened()
        print(rval)
        while rval: # 循环读取视频
            c = c + 1
            rval, frame = video.read()
            
            if(c%30==0):
                if rval:
                    # # undistort-------------------------------------------
                    # h,w = frame.shape[:2]
                    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                    # frame = cv2.undistort(frame,mtx,dist,None,newcameramtx)
                    # x,y,w,h = roi
                    # frame = frame[y:y+h,x:x+w]
                    # # ----------------------------------------------------

                    frame_dir = parent_dir + '/' + 'frame data/' + base_name + '/'
                    cv2.imwrite(frame_dir + base_name[:-4] + str(c) + '.jpg', frame)
                    # cv2.imwrite(frame_dir + '%06d.jpg'%(c), frame)
                    cv2.waitKey(1)
                else:
                    break
        video.release()
        print('save '+ base_name +' success')
        messagebox.showinfo(title='success', message='提取帧结束')

    """
    将视频提取成帧，方便之后的处理，视频都存储在"./video data"
    """
    def create_video_frame(self, event):
        self.videoDir = filedialog.askopenfilename(title = "Select Video")
        if not self.videoDir:
            return None
        base_name = os.path.basename(self.videoDir) # 读取当前文件名
        parent_dir = os.path.dirname(os.path.dirname(self.videoDir))  # 读取当前文件的父级目录
        # print(base_name, parent_dir)
        isExists=os.path.exists(parent_dir + '/' + 'frame data/' + base_name)  # 判断当前路径是否存在
        if not isExists:    # 当前这个视频还没有被提取帧
            kind = filetype.guess(self.videoDir)
            file_kind = os.path.dirname(kind.mime)  # 获得当前选择的文件类型
            if(file_kind == 'video'):
                print('create new folder: ' + parent_dir + '/' + 'frame data/' + base_name)
                os.makedirs(parent_dir + '/' + 'frame data/' + base_name)
                self.video_to_frame(self.videoDir)
            else:
                messagebox.showinfo(title='提示', message='请选择合适的视频文件')
        else:      # 当前这个视频已经被提取帧了，提示是否重新提取
            answer = messagebox.askquestion(title='提示', message='该文件已存在，是否重新提取帧')
            if (answer == 'yes'):  # 如果要重新提取
                print('Re-extraction frame: ' + parent_dir + '/' + 'frame data/' + base_name)
                self.video_to_frame(self.videoDir)

    """
    读取某帧图片，且可以定位该帧的位置，即可以从中间某一帧开始，往前往后继续手动标注
    """
    def open_image(self, event):
        self.filename = filedialog.askopenfilename(title="Select Image", 
                                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if not self.filename:
            return None
        # print(self.filename)
        self.globalfiledir = self.filename  # 用于函数：self.save_to_csvs
        basename = os.path.basename(self.filename)
        parent_dir = os.path.dirname(self.filename)
        base_name = os.path.basename(os.path.dirname(self.filename)) # 读取当前文件名
        lengh = len(base_name) - 4
        # print(basename)
        # print(parent_dir)
        # print(base_name)
        self.cur = int(basename[lengh:-4]) - 1
        self.imageList = os.listdir(parent_dir)
        self.imageList.sort(key = lambda x: int(x[:-4]))
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = parent_dir
        self.first_load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur], event)

    """
    打开一个存储一个视频所有frame的文件夹，及从第一帧开始手动标注
    """
    def open_image_dir(self, event):
        self.imageDir = filedialog.askdirectory(title="Select Image Directory")
        if not self.imageDir:
            return None
        # print(self.imageDir)
        self.globalfiledir = self.imageDir  # 用于函数：self.save_to_csv
        self.imageList = os.listdir(self.imageDir)
        self.imageList.sort(key = lambda x: int(x[:-4])) #sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        self.first_load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur], event)

    """
    最开始加载图片时，需要将相应的数据集也加载进来
    """
    def first_load_image(self, file, event):
        base_name = os.path.basename(os.path.dirname(file)) # 读取当前文件名
        # print(base_name)
        isExists=os.path.exists('./csv/' + base_name + '.csv')  # 判断当前文件是否存在
        if not isExists:
            messagebox.showinfo(title='提示', message='不存在相应的数据文件')
            self.img_message = pd.DataFrame(columns=['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'group']) 
        else:
            self.img_message = pd.read_csv('./csv/' + base_name + '.csv') # 打开相应的数据集文件
            col_name = self.img_message.columns.values
            if (len(col_name) == 6):
                self.img_message['group_id'] = 0      # 添加一列groupid, 0表示单个行人

        self.load_image(file, event)
        

    """
    reload the new track id
    """
    def reload_track_id(self):
        if(self.img_message.shape[0] > 0):
            # 找到trackid的最大值，以防出现没有检测到的情况，可以添加新的检测框
            self.max_track_id = self.img_message.loc[:,"track_id"].max()
            max_group_id = self.img_message.loc[:,"group_id"].max()
        else:
            self.max_track_id = int(0)
            max_group_id = int(0)
            
        alltrackid = self.img_message.loc[:,"track_id"].values
        # print(alltrackid)
        new_track_id = self.max_track_id + 1
        for i in range(1, self.max_track_id + 1):
            if i not in alltrackid:
                new_track_id = i
                break

        new_group_id = max_group_id + 1
        return new_track_id, int(new_group_id)

    """
    把当前的图片读取出来
    """
    def load_image(self, file, event):
        new_track_id, new_group_id = self.reload_track_id()
        self.newtrackidlabel["text"] = "New track Id (suggest: " + str(new_track_id)+')'  # 把当前最大的trackid显示到窗口上，作为提示
        self.labelGroupId["text"] = "New group Id (suggest: " + str(new_group_id)+')'
        # global img_message
        self.img = Image.open(file)
        self.imgfile = self.imageList[self.cur]
        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='  ||  Image Number: %d / %d' % (self.imageCur, self.imageTotal))
        # Resize to Pascal VOC format
        w, h = self.img.size
        if w >= h:
            baseW = 800
            wpercent = (baseW / float(w)) # wpercent = 0.625
            hsize = int((float(h) * float(wpercent)))
            self.img = self.img.resize((baseW, hsize), Image.BICUBIC)
        else:
            baseH = 450
            wpercent = (baseH / float(h))
            wsize = int((float(w) * float(wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.BICUBIC)
        self.rate = float(float(w)/baseW) # 从(baseW,baseH)→(w,h)需要扩大的倍数
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.clear_bbox()

        base_name = os.path.basename(os.path.dirname(file)) # 读取当前文件名
        lengh = len(base_name) - 4
        # 在这里读取图片的同时把图片里的检测框也读出来！！！！！
        self.frameId = self.imgfile[lengh-len(self.imgfile):-4]
        # print(self.frameId)
        if self.img_message.empty:
            return
        choosebyFrameId = self.img_message[(self.img_message.frame_id == int(self.frameId))]
        for i in range(len(choosebyFrameId)):
            x = int(((choosebyFrameId.iloc[[i],[2]]).values[0])[0] / self.rate)
            y = int(((choosebyFrameId.iloc[[i],[3]]).values[0])[0] / self.rate)
            w = int(((choosebyFrameId.iloc[[i],[4]]).values[0])[0] / self.rate)
            h = int(((choosebyFrameId.iloc[[i],[5]]).values[0])[0] / self.rate)
            track_id = ((choosebyFrameId.iloc[[i],[1]]).values[0])[0]

            self.STATE['x'], self.STATE['y'] = x, y
            event.x = x+w
            event.y = y+h
            self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'], event.x, event.y, width=2, outline='red')
            self.boxtrackid[self.bboxId] = track_id
            self.mouse_release(event)
        self.draw_groupid()

    """
    把当前处理好的结果存储下来，并打开下一张图片和加载相应的检测信息
    """
    def open_next(self, event):
        if self.cur < len(self.imageList) - 1:
            self.save()
            self.clear_bbox()
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur], event)
        else:
            self.save()
            messagebox.showinfo(title='',message='the last picture')    #提示信息对话窗
            return
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    """
    把当前处理好的结果存储下来，并打开上一张图片和加载相应的检测信息
    """
    def open_previous(self, event):
        if self.cur > 0:
            self.save()
            self.clear_bbox()
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur], event)
        else:
            messagebox.showinfo(title='',message='the first picture')
            return 
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    """
    存储结果
    """
    def save(self):
        for idx, item in enumerate(self.bboxList):
            
            choosebyFrameandTrack = self.img_message[(self.img_message.frame_id == int(self.frameId)) & 
                                                (self.img_message.track_id == int(self.boxtrackid[self.bboxIdList[idx]]))]
            
            # 如果该检测框在dataframe中不存在
            if choosebyFrameandTrack.empty: 
                trackid = int(self.boxtrackid[self.bboxIdList[idx]])
                # 从dataframe中取出当前的frame id的一行
                choosebyFrame = self.img_message[(self.img_message.frame_id == int(self.frameId))]  # 先根据frameid筛选一下img_message
                
                # 如果在该frame下没有检测框，需要找到比当前frameid大和小的位置，把新一行放进去
                if choosebyFrame.empty:
                    chooseBigFrameId = self.img_message[self.img_message.frame_id > int(self.frameId)] # 筛选出比当前frameid大的所有dataframe
                    # 如果没有比当前frameid大的数据，即该帧在最后，插在最后
                    if chooseBigFrameId.empty:
                        # time = self.img_message.time.tolist()[0]                          # 得到时间
                        # lastframeid = self.img_message.frame_id.tolist()[0]               # 得到最后一行的frameid
                        # curframetime= time + 39*(int(self.frameId)-lastframeid)
                        x = list(self.bboxList[idx])[0] * self.rate
                        y = list(self.bboxList[idx])[1] * self.rate
                        w = (list(self.bboxList[idx])[2] - list(self.bboxList[idx])[0]) * self.rate
                        h = (list(self.bboxList[idx])[3] - list(self.bboxList[idx])[1]) * self.rate
                        # get_groupid = self.newgroupidentry.get()
                        # if get_groupid:
                        #     groupid = int(get_groupid)
                        # else:
                        #     groupid = None
                        insertRow = pd.DataFrame([[int(self.frameId), trackid, x , y, w, h, 0]],
                                                columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'group_id'])
                        self.img_message = self.img_message.append(insertRow, ignore_index=True)

                    # 有比当前frameid大的数据，即该帧在开头或者中间，按照顺序插入
                    else:
                        bigerIdrow = int(chooseBigFrameId.index.tolist()[0])                               # 得到行号
                        
                        # time = chooseBigFrameId.time.tolist()[0]                                           # 得到时间
                        # nextframeid = chooseBigFrameId.frame_id.tolist()[0]                                # 得到frameid
                        # curframetime = time - 39*(nextframeid - int(self.frameId))
                        x = list(self.bboxList[idx])[0] * self.rate
                        y = list(self.bboxList[idx])[1] * self.rate
                        w = (list(self.bboxList[idx])[2] - list(self.bboxList[idx])[0]) * self.rate
                        h = (list(self.bboxList[idx])[3] - list(self.bboxList[idx])[1]) * self.rate
                        # get_groupid = self.newgroupidentry.get()
                        # if get_groupid:
                        #     groupid = int(get_groupid)
                        # else:
                        #     groupid = None
                        insertRow = pd.DataFrame([[int(self.frameId), trackid, x , y, w, h, 0]],
                                                columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'group_id'])
                        above = self.img_message.loc[:bigerIdrow-1]
                        below = self.img_message.loc[bigerIdrow:]
                        self.img_message = above.append(insertRow, ignore_index=True).append(below, ignore_index=True)
                    
                # 如果在该frame下有检测框，则找到该frame下的track id的最大值，然后插在该行的下面
                else:
                    # 找到该frame下的trackid最大值的行号，然后把新的检测框放到改行之后---------------------------------------
                    maxIdOfTheFrame = choosebyFrame.loc[:,"track_id"].max()                   # 然后找到trackid的最大值
                    choosebyFrameandMaxid = choosebyFrame[(choosebyFrame.track_id == maxIdOfTheFrame)] # 把最大值行筛选出来
                    row = int(choosebyFrameandMaxid.index.tolist()[0])                        # 得到行号
                    # time = choosebyFrame.time.tolist()[0]                                     # 一个frame下的检测框时间是相同的
                    # ---------------------------------------------------------------------------------------------------
                    x = list(self.bboxList[idx])[0] * self.rate
                    y = list(self.bboxList[idx])[1] * self.rate
                    w = (list(self.bboxList[idx])[2] - list(self.bboxList[idx])[0]) * self.rate
                    h = (list(self.bboxList[idx])[3] - list(self.bboxList[idx])[1]) * self.rate
                    # get_groupid = self.newgroupidentry.get()
                    # if get_groupid:
                    #     groupid = int(get_groupid)
                    # else:
                    #     groupid = None
                    insertRow = pd.DataFrame([[int(self.frameId), trackid, x , y, w, h, 0]],
                                            columns = ['frame_id','track_id','x','y','w','h','group_id'])
                    above = self.img_message.loc[:row]
                    below = self.img_message.loc[row+1:]
                    self.img_message = above.append(insertRow,ignore_index=True).append(below,ignore_index=True)
            
            # 如果该检测框在dataframe中存在
            else:  
                rowNumber = int(choosebyFrameandTrack.index.tolist()[0])
                self.img_message.iloc[[rowNumber], [2]] = list(self.bboxList[idx])[0] * self.rate # 左上角x
                self.img_message.iloc[[rowNumber], [3]] = list(self.bboxList[idx])[1] * self.rate # 左上角y
                self.img_message.iloc[[rowNumber], [4]] = (list(self.bboxList[idx])[2] - list(self.bboxList[idx])[0]) * self.rate # 框的宽
                self.img_message.iloc[[rowNumber], [5]] = (list(self.bboxList[idx])[3] - list(self.bboxList[idx])[1]) * self.rate # 框的宽

    """
    把组成同一个group的检测框框起来
    """
    def draw_groupid(self):
        self.objectListgroup.delete(0, len(self.bboxIdgroupList))
        self.canvas.delete(self.bboxIdgroup)
        self.bboxIdgroupList = []
        
        if self.img_message.empty:
            return
        choosebyFrameId = self.img_message[(self.img_message.frame_id == int(self.frameId))]
        list_choosebyFrameId = choosebyFrameId.index.tolist()
        groupid = []
        for i in list_choosebyFrameId:
            temp = ((self.img_message.iloc[[i],[6]]).values[0])[0]
            if temp!=0 and temp not in groupid:
                groupid.append(temp)

        for i in groupid:
            choosebyFrameandGroup = self.img_message[(self.img_message.frame_id == int(self.frameId)) &
                                                     (self.img_message.group_id == int(i))]
            if len(choosebyFrameandGroup) == 1:
                continue
            x1 = int(((choosebyFrameandGroup.iloc[[0],[2]]).values[0])[0] / self.rate)
            y1 = int(((choosebyFrameandGroup.iloc[[0],[3]]).values[0])[0] / self.rate)
            w1 = int(((choosebyFrameandGroup.iloc[[0],[4]]).values[0])[0] / self.rate)
            h1 = int(((choosebyFrameandGroup.iloc[[0],[5]]).values[0])[0] / self.rate)
            x2 = x1 + w1
            y2 = y1 + h1
            strtrackid = ''
            for j in range(len(choosebyFrameandGroup)):
                strtrackid = strtrackid + ' ' + str(((choosebyFrameandGroup.iloc[[j],[1]]).values[0])[0])
                if (int(((choosebyFrameandGroup.iloc[[j],[2]]).values[0])[0] / self.rate) < x1):
                    x1 = int(((choosebyFrameandGroup.iloc[[j],[2]]).values[0])[0] / self.rate)
                if (int(((choosebyFrameandGroup.iloc[[j],[3]]).values[0])[0] / self.rate) < y1):
                    y1 = int(((choosebyFrameandGroup.iloc[[j],[3]]).values[0])[0] / self.rate)
                if (int(((choosebyFrameandGroup.iloc[[j],[2]]).values[0])[0] / self.rate)+int(((choosebyFrameandGroup.iloc[[j],[4]]).values[0])[0] / self.rate) > x2):
                    x2 = int(((choosebyFrameandGroup.iloc[[j],[2]]).values[0])[0] / self.rate)+int(((choosebyFrameandGroup.iloc[[j],[4]]).values[0])[0] / self.rate)
                if (int(((choosebyFrameandGroup.iloc[[j],[3]]).values[0])[0] / self.rate)+int(((choosebyFrameandGroup.iloc[[j],[5]]).values[0])[0] / self.rate) > y2):
                    y2 = int(((choosebyFrameandGroup.iloc[[j],[3]]).values[0])[0] / self.rate)+int(((choosebyFrameandGroup.iloc[[j],[5]]).values[0])[0] / self.rate)
            
            self.bboxIdgroup = self.canvas.create_rectangle(x1-5, y1-5, x2+5, y2+5, width=2, outline='yellow')
            self.bboxIdgroupList.append(self.bboxIdgroup)
            self.objectListgroup.insert(END, strtrackid + ': ' + str(i))
            self.objectListgroup.itemconfig(len(self.bboxIdgroupList) - 1, fg='red')
            self.boxgroupid[self.bboxIdgroup] = int(i)
        # print(self.bboxIdgroupList)

    """
    从输入框中读入想要成组的track id和组的编号group id，更新img_message并在图上画出group的框
    """
    def button_groupid(self):
        entryTrackId = self.entryTrackId.get()
        entryGroupId = self.entryGroupId.get()
        if entryTrackId and entryGroupId:
            list_trackid = entryTrackId.split()
            for i in list_trackid:
                # choosebyTrack = self.img_message[(self.img_message.track_id == int(i))]
                # list_choosebyTrack = choosebyTrack.index.tolist()
                choosebyTrackFrame = self.img_message[(self.img_message.track_id == int(i)) & 
                                                      (self.img_message.frame_id == int(self.frameId))]
                list_choosebyTrack = choosebyTrackFrame.index.tolist()
                for index in list_choosebyTrack:
                    self.img_message.iloc[[index],[6]] = int(entryGroupId)
            self.draw_groupid()
        else:
            messagebox.showinfo(title='',message='TrackId & GroupId\ncannot be empty')
            return
        # print(self.img_message)

    """
    鼠标在图片上点击时可以获取位置
    """
    def mouse_click(self, event):
        # Check if Updating BBox
        # find_enclosed(x1, y1, x2, y2)-- 返回完全包含在限定矩形内所有画布对象的 ID
        if self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5):
            self.EDIT = True
            self.editPointId = int(self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5)[0])
        else:
            self.EDIT = False

        # Set the initial point
        if self.EDIT:
            idx = self.bboxPointList.index(self.editPointId)
            self.editbboxId = self.bboxIdList[math.floor(idx/4.0)]
            self.bboxId = self.editbboxId
            pidx = self.bboxIdList.index(self.editbboxId)
            pidx = pidx * 4
            self.o1 = self.bboxPointList[pidx]
            self.o2 = self.bboxPointList[pidx + 1]
            self.o3 = self.bboxPointList[pidx + 2]
            self.o4 = self.bboxPointList[pidx + 3]
            a = 0
            b = 0
            c = 0
            d = 0
            if self.editPointId == self.o1:
                a, b, c, d = self.canvas.coords(self.o3)
            elif self.editPointId == self.o2:
                a, b, c, d = self.canvas.coords(self.o4)
            elif self.editPointId == self.o3:
                a, b, c, d = self.canvas.coords(self.o1)
            elif self.editPointId == self.o4:
                a, b, c, d = self.canvas.coords(self.o2)
            self.STATE['x'], self.STATE['y'] = int((a+c)/2), int((b+d)/2)
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y

    """
    拖动鼠标事件：画出矩形框
    """
    def mouse_drag(self, event):
        self.mouse_move(event)
        # print(self.bboxId)
        track_id = 0
        if self.bboxId:
            self.currBboxColor = self.canvas.itemcget(self.bboxId, "outline")
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.o1)
            self.canvas.delete(self.o2)
            self.canvas.delete(self.o3)
            self.canvas.delete(self.o4)
            track_id = self.boxtrackid[self.bboxId]
            self.canvas.delete("box"+str(self.boxtrackid[self.bboxId]))
            del self.boxtrackid[self.bboxId]
        if self.EDIT:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)
            self.boxtrackid[self.bboxId] = track_id
        else:
            self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)
            self.boxtrackid[self.bboxId] = track_id

    """
    鼠标指的地方可以在窗口的左边有个放大的处理
    """
    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x*self.rate, event.y*self.rate))
        self.zoom_view(event)    # 放大局部
        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
    
    """
    释放鼠标，如果这个矩形框在图片中是存在的，则更新它的对角位置，如果是新的矩形框，则在右上角显示
    """
    def mouse_release(self, event):
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        if self.EDIT:
            self.update_bbox()
            self.EDIT = False
        # print(self.STATE['x'],self.STATE['y'],event.x,event.y)
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
        var = ''
        if x1 != x2:  
            # 如果不是一个点，即必须画的是矩形框
            # 则把画的矩形框的左上角右下角的坐标值存储下来
            self.bboxList.append((x1, y1, x2, y2))
            o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
            o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
            o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
            o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
            self.bboxPointList.append(o1)
            self.bboxPointList.append(o2)
            self.bboxPointList.append(o3)
            self.bboxPointList.append(o4)
            self.bboxIdList.append(self.bboxId)
            self.objectLabelList.append(str(self.currLabel))

            if self.boxtrackid.__contains__(self.bboxId) and self.boxtrackid[self.bboxId]!=0 :
                # 如果字典结构boxtrackid中存在这个bboxid
                # 则在框的左上角加上trackid
                var = str(self.boxtrackid[self.bboxId])
                self.canvas.create_text(x1,y1-14,text = var,fill='red',font=('Arial',18),tags = "box" + var)
            else:
                # 如果字典结构boxtrackid中不存在这个bboxid
                # 则从newtrackid这个entry中获得新id
                newtrackid = self.newtrackid.get()
                if newtrackid:
                    # 如果这个id不为空
                    # 则把这个bboxid和相应的trackid存到字典序列boxtrackid中
                    # 并且在左上角画出相应的trackid
                    self.boxtrackid[self.bboxId]=newtrackid
                    var = str(newtrackid)
                    self.canvas.create_text(x1,y1-14,text = var,fill='red',font=('Arial',18),tags = "box" + var)
                else:
                    # 如果获得的id为空
                    # 则这种情况不能出现
                    # 则把已经画出来的矩形框、trackid等删除，并return
                    self.boxtrackid[self.bboxId]= newtrackid
                    self.bboxId = None
                    self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1*self.rate, y1*self.rate, x2*self.rate, y2*self.rate) + ': ' + var)
                    self.newtrackid.delete(0,END)
                    self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=self.currBboxColor)
                    self.currLabel = None
                    self.del_bbox()
                    self.save()
                    # print(self.boxtrackid)
                    return
            # bboxId存在或者得到的新id不为空则把相应的数据存储下来，并显示到右上角的listbox中
            self.bboxId = None
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1*self.rate, y1*self.rate, x2*self.rate, y2*self.rate) + ': ' + var)
            self.newtrackid.delete(0,END)
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=self.currBboxColor)
            self.currLabel = None
        self.save()
        self.draw_groupid()
        # print(self.boxtrackid)

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 25), (event.y - 25), (event.x + 25), (event.y + 25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 75, 150, 75, width=2)
            vl = self.zoomcanvas.create_line(75, 0, 75, 150, width=2)
        except:
            pass

    def update_bbox(self):
        idx = self.bboxIdList.index(self.editbboxId)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)
        self.currLabel = self.objectLabelList[idx]
        self.objectLabelList.pop(idx)
        idx = idx*4
        self.canvas.delete(self.bboxPointList[idx])
        self.canvas.delete(self.bboxPointList[idx+1])
        self.canvas.delete(self.bboxPointList[idx+2])
        self.canvas.delete(self.bboxPointList[idx+3])
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)

    def cancel_bbox(self, event):
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    """
    把“List of Objects”里选中的track矩形框信息删除，并更新dataframe信息
    """
    def del_bbox(self):
        # global img_message
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            idx = len(self.bboxIdList) - 1
        else:
            idx = int(sel[0])
        # 删除矩形框和和四个角
        self.canvas.delete(self.bboxIdList[idx])
        self.canvas.delete(self.bboxPointList[idx * 4])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 1])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 2])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 3])
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        bboxId = self.bboxIdList.pop(idx)
        if self.boxtrackid[bboxId]:
            # 如果从bboxlist中删除一个有trackid的矩形框，则也要从相应的img_message的dataframe中删除
            # 先找到相应frameId、trackid的行号，然后从打他frame中drop掉
            deleteDataframe = self.img_message[(self.img_message.frame_id == int(self.frameId)) & 
                                          (self.img_message.track_id == int(self.boxtrackid[bboxId]))]
            row = int(deleteDataframe.index.tolist()[0])
            self.img_message.drop(row,axis=0,inplace=True) # 删除该行
            self.img_message = self.img_message.reset_index(drop=True)
        # 把矩形框左上角的trackid文本删除
        self.canvas.delete("box"+str(self.boxtrackid[bboxId]))
        del self.boxtrackid[bboxId] 
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)
    
    """
    把“List of Objects”里选中的track矩形框信息删除，不更新dataframe信息
    """
    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        for idx in range(len(self.boxtrackid)):
            bboxId = self.bboxIdList[idx]
            self.canvas.delete("box"+str(self.boxtrackid[bboxId]))
            del self.boxtrackid[bboxId]
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []
        self.boxtrackid = dict()

    """
    把“List of Objects”里的所有的track矩形框信息删除，并更新dataframe信息
    """
    def clear_all(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        for idx in range(len(self.boxtrackid)):
            bboxId = self.bboxIdList[idx]
            if self.boxtrackid[bboxId]:
                # 如果从bboxlist中删除一个有trackid的矩形框，则也要从相应的img_message的dataframe中删除
                # 先找到相应frameId、trackid的行号，然后从打他frame中drop掉
                deleteDataframe = self.img_message[(self.img_message.frame_id == int(self.frameId)) & 
                                              (self.img_message.track_id == int(self.boxtrackid[bboxId]))]
                row = int(deleteDataframe.index.tolist()[0])
                self.img_message.drop(row,axis=0,inplace=True) # 删除该行
                self.img_message = self.img_message.reset_index(drop=True)
            self.canvas.delete("box"+str(self.boxtrackid[bboxId]))
            del self.boxtrackid[bboxId]
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []
        self.boxtrackid = dict()

    """
    把“List of Groups”里的选中的group信息删除，并更新dataframe信息
    """
    def del_bboxgroup(self):
        sel = self.objectListgroup.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        # print(self.bboxIdgroupList[idx])
        self.canvas.delete(self.bboxIdgroupList[idx])
        bboxgroupId = self.bboxIdgroupList.pop(idx)
        if self.boxgroupid[bboxgroupId]:
            # 如果从bboxlist中删除一个有trackid的矩形框，则也要从相应的img_message的dataframe中删除
            # 先找到相应frameId、trackid的行号，然后从打他frame中drop掉
            alterDataframe = self.img_message[(self.img_message.frame_id == int(self.frameId)) & 
                                          (self.img_message.group_id == int(self.boxgroupid[bboxgroupId]))]
            row = alterDataframe.index.tolist()
            for index in row:
                self.img_message.iloc[[index],[6]] = 0
        del self.boxgroupid[bboxgroupId]
        self.objectListgroup.delete(idx)
        print(self.bboxIdgroupList)
    
    """
    把“List of Groups”里的所有group信息删除，并更新dataframe信息
    """
    def clear_allgroup(self):
        for idx in range(len(self.bboxIdgroupList)):
            self.canvas.delete(self.bboxIdgroupList[idx])
        
        for idx in range(len(self.bboxIdgroupList)):
            bboxgroupId = self.bboxIdgroupList[idx]
            if self.boxgroupid[bboxgroupId]:
                # 如果从bboxlist中删除一个有trackid的矩形框，则也要从相应的img_message的dataframe中删除
                # 先找到相应frameId、trackid的行号，然后从打他frame中drop掉
                alterDataframe = self.img_message[(self.img_message.frame_id == int(self.frameId)) & 
                                            (self.img_message.group_id == int(self.boxgroupid[bboxgroupId]))]
                row = alterDataframe.index.tolist()
                for index in row:
                    self.img_message.iloc[[index],[6]] = 0
            del self.boxgroupid[bboxgroupId]
        self.objectListgroup.delete(idx)
        self.objectLabelList = []
        self.boxgroupid = dict()
        print(self.bboxIdgroupList)

    """
    把标定好的结果写入到.csv文件中
    """
    def save_to_csv(self):
        lastdir = self.globalfiledir[-3:]
        if lastdir == 'jpg':
            basename = os.path.basename(os.path.dirname(self.globalfiledir))
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.globalfiledir)))
            
        else:
            basename = os.path.basename(self.globalfiledir)
            parent_dir = os.path.dirname(os.path.dirname(self.globalfiledir))
        
        self.img_message.to_csv(parent_dir + '/' + 'csv/out_' + basename + '.csv', index=0)
        print("save success")
        messagebox.showinfo(title='提示', message='save success')
        print(self.img_message)

if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon1.png')
    root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()