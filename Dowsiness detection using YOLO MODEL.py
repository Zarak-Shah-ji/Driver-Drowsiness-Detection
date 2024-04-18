#!/usr/bin/env python
# coding: utf-8

# # Installing dependencies 
 
# Installing pytorch  
 
# In[5]: 
  
 
get_ipython().system('pip3 install torch torchvision torchaudio')


# Cloning the yolo5 model for detection with all requirements   

# In[1]:



get_ipython().system('git clone https://github.com/ultralytics/yolov5 ')


# In[7]:


import os
os.chdir('/Users/zshahjee/Downloads/yolov5')
get_ipython().system('pip install -r requirements.txt')


# In[8]:


get_ipython().system('python -m pip install pefile')
 

# In[13]:


get_ipython().system('python fixNvPe.py --input = C:\\Users\\ms714139\\Anaconda3\\Lib\\site-packages\\torch\\lib\\*.dll')

get_ipython().system('python fixNvPe.py --input=C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\torch\\lib\\*.dll')


# # Importing all dependencies

# In[1]:


import torch 
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[2]:


import os    # used for kernel dying issue by using imshow of matplotlib
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# # Loading Model

# In[3]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[4]:


model


# # Make detections with images

# In[5]:


#img = 'https://ultralytics.com/images/zidane.jpg'
img = 'https://www.irishnews.com/picturesarchive/irishnews/irishnews/2017/05/11/161016492-04b25bf7-73f0-473d-81fb-6a35c7eeb0cf.jpg'


# In[6]:


results = model(img)
results.print()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[18]:


#plt.imshow(results.render())


# # Dimensions and Shape of the image

# In[8]:


results


# In[10]:


results.xyxy


# In[9]:


results.show() #opening image in preview 


# In[11]:


results.render()   # drwaing the image array


# In[17]:


#real shape of model
np.array(results.render()).shape


# In[19]:


#to render the image using matplotlib without the error of Invalid shape (1, 720, 1280, 3) for image data
#we squeeze the image
np.squeeze(results.render()).shape


# # Real time detections

# In[8]:


#for front camera i have 1 as the number in my surface and 0 for rear camera
cap = cv2.VideoCapture(1)  #accessing our webcam #change the digit from 1, 0,2 for diff devices 
while cap.isOpened():
    ret, frame = cap.read()  #return value and image frame from webcam
    
    #Make detections
    results = model(frame)
    cv2.imshow('YOLO', frame) #the window name at the top = YOLO
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # YOLO Detecting different classes

# In[11]:


#to capture through video just keep the video in the same folder as where the yolo5 is
#the video will play slow on surface due to lack of gpu

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('car2.mp4')  #accessing our webcam #change the digit from 1, 0,2 for diff devices 
while cap.isOpened():
    ret, frame = cap.read()  #return value and image frame from webcam
    
    #Make detections
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render())) #the window name at the top = YOLO
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(10)


# # Training from scratch for drowsiness

# In[3]:


import uuid  #creating unique identifier # used to name the images
import os   #for working with our file paths 
import time  #to take a break between each image we are collecting about for the dataset


# In[5]:


IMAGES_PATH = os.path.join('data','images')  #/data/images
#folder = data for all images created , subfolder = images 
labels = ['awake','drowsy']  #classifying images
number_imgs = 20


# # Training loop for image data collection

# In[7]:


cap  = cv2.VideoCapture(0)

#loop through labels
for label in labels:
    print('Collecting images for {}'.format(label)) #so that we can see when transitioning through images
    time.sleep(5)  #sleep or wait for 5 seconds when transitioning between image for other class
    
    #Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        #webcam feed
        ret, frame = cap.read()   # read the feed from the web cam and store it in given vars
        
        #Naming image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        #Writes out image to file
        cv2.imwrite(imgname,frame)
        
        # Render to screen
        cv2.imshow('Image Collection', frame)
        
        # 2 sec delay between diff captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
    


# # Labelling collected image dataset for "awake" and "drowsy" class

# In[1]:


get_ipython().system('git clone https://github.com/tzutalin/labelImg')


# Installing below dependencies of labelImg for Mac OS

# In[2]:


get_ipython().system('brew install qt  # Install qt-5.x.x by Homebrew for Mac OS')
get_ipython().system('brew install libxml2')


# In[ ]:


#Installing labelImg  
#installed using terminal with commands :
#pip3 install labelImg
#open labelimg using :
#labelImg  cmd in terminal


# # Making the model train on our image dataset

# In[ ]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt')


# # Loading Custom Model

# In[4]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)


# In[5]:


img = os.path.join('data', 'images', 'awake.a465b15a-cb30-11ec-afc5-2af8bc7a8036.jpg')


# In[6]:


results = model(img)


# In[7]:


results.print()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[9]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(10)


# In[ ]:




