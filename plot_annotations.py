import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib import animation, rc
rc('animation', html='jshtml')

img_dir='//media/genadiy/C/data/VisDrone/images/train'
txt_dir='/media/genadiy/C/data/VisDrone/labels/train'

impaths=[]
files=os.listdir(img_dir)
for item in files:
    impaths+=[os.path.join(img_dir,item)]
impaths.sort()

images0=[]
for i in tqdm(range(len(impaths)//4)):
    images0+=[cv2.imread(impaths[i])]
    
txtpaths=[]
texts=os.listdir(txt_dir)
for item in texts:
    txtpaths+=[os.path.join(txt_dir,item)]
txtpaths.sort()
impaths=impaths[0:16]
txtpaths=txtpaths[0:16]
boxdata=[]
boxfile=[]
for i in range(len(txtpaths)):#len(txtpaths)
    path=txtpaths[i]
    dfi=pd.read_csv(path,header=None)
    #display(dfi)
    b=[]
    for j in range(len(dfi)):
       b+=[dfi.iloc[j,0].split(' ')] 
    boxdata+=[b] 
    boxfile+=[path[0:-4].split('/')[-1]]

def draw_box(num0):
    
    impath=impaths[num0]
    #print(impath)
    image=cv2.imread(impath)

    H,W=image.shape[0],image.shape[1]
    file=impath[0:-4].split('/')[-1]
    #print(H,W)
    
    box=boxdata[num0]

    for i in range(len(box)):#len(box)
        x=float(box[i][1])
        y=float(box[i][2])
        w=float(box[i][3])
        h=float(box[i][4])
        #print(x,y,h,w)
        
        x0=int((x-w/2)*W)
        y0=int((y-h/2)*H)
        x1=int((x+w/2)*W)
        y1=int((y+h/2)*H)
        #print((x0,y0),(x1,y1))

        cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0),2)                       
        
    cv2.imwrite('/kaggle/working/rimages/'+file+'.png',image) 
    plt.figure(figsize=(15,15))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show() 
        
    return image

images1=[]
for i in tqdm(range(len(impaths))):
    images1+=[draw_box(i)]