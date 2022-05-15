import jcopdl
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from jcopdl.utils.dataloader import MultilabelDataset
from PIL import Image
bs = 16
crop_size = 224

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(crop_size, scale = (0.9, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
])
train_set = datasets.ImageFolder("oilspill_dataset_fix/train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=1)

test_set = datasets.ImageFolder("oilspill_dataset_fix/test/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

label2cat = train_set.classes

from torchvision.models import mobilenet_v2

mnet = mobilenet_v2(pretrained = True) #akan download mobilenet weightnya dulu

#Cara untuk ngefreeze
for param in mnet.parameters():
    param.requires_grad = False

class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.LogSoftmax()
        )
        
    def forward(self,x):
        return self.mnet(x)
    
    def freeze(self):
        for param in mnet.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in mnet.parameters():
            param.requires_grad = True

# Parameter apa saja yang ingin disimpan, biasanya crop size 224 x 224
config = set_config({
    "output_size" : len(train_set.classes),
    "batch_size" : bs,
    "crop_size" : crop_size
})


model = CustomMobilenetV2(config.output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, early_stop_patience = 5, outdir="model")


#best_model = torch.load("model/weights_best.pth")
#model.load_state_dict(best_model)


def image_loader(image_name):
    image = test_transform(image_name).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

def predict_spill(x):
    with torch.no_grad():
        model.eval()
        output = model (x)
        #print(f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0]))))
        return str(label2cat[output.argmax(1).item()]) + ' with probability ' + str("{:.2f}".format(max(np.exp(output)[0])))


import time
import cv2
# the value can be 0 or -1
# cv2.CAP_DSHOW to prevent error


# importing the required packages
import pyautogui
import cv2
import numpy as np
  
# Specify resolution
resolution = (1920, 1080)
codec = cv2.VideoWriter_fourcc(*"XVID")
filename = "Recording.avi"
fps = 60.0
out = cv2.VideoWriter(filename, codec, fps, resolution)
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 480, 270)
a_w = 640/1920
a_h = 480/1080

def draw(title):
    font = cv2.FONT_ITALIC
    if title.split(" ")[0] == "oilspill":
        cv2.putText(frame,title,(20,int(50)), font, 1,(0,0,255),2,cv2.LINE_AA)
    if title.split(" ")[0] == "nospill":
        cv2.putText(frame,title,(20,int(50)), font, 1,(0,255,0),2,cv2.LINE_AA)
cap = cv2.VideoCapture(0)
while True:
    # Take screenshot using PyAutoGUI from monitor 1
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
    # Write it to the output file
    out.write(frame)
    
    frame_to_predict = cv2.resize(frame, None, fx= a_w, fy=a_h, interpolation =cv2.INTER_LINEAR)
    x = image_loader(Image.fromarray(frame))
    title = predict_spill(x)
    cv2.rectangle(frame, (0,0), (600, 80), (0,0,0), -1)
    draw(title)
    cv2.imshow("Live", frame) #show the video
      
    # Stop recording when we press 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() #release memory
cv2.destroyAllWindows()