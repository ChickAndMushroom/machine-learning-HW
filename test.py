import csv
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from math import sqrt
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import time
import random


test_file_name='sampleSubmission.csv' #借用处理后的sampleSubmission读取测试集数据的name

path2="test" #测试集数据所在的文件夹
learning_rate=1e-3
NUM_EPOCHS=50
is_mixup = True
batch_size=64
rot_flag=True

# preprocessing


normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

class xnet(nn.Module):
	def __init__(self):
		super(xnet, self).__init__()

		self.features = nn.Sequential(

			nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
			nn.BatchNorm3d(16),

			nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
			nn.BatchNorm3d(32),

			nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
			nn.BatchNorm3d(64),




# state size. 64 x 4 x 4 x 4 

		)

		self.ReLU = nn.ReLU(inplace=True)
		self.Dropout = nn.Dropout(p=0.5)

		self.fc6 = nn.Linear(64 * 4 * 4 * 4, 20)
		self.BN=nn.BatchNorm1d(20)
		self.fc7 = nn.Linear(20, 2)


		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, input):
		out = self.features(input)
		out = out.view(out.size(0), -1)
		out = self.fc6(out)
		out = self.BN(out)
		out = self.ReLU(out)
		out = self.Dropout(out)
		out = self.fc7(out)

		return F.softmax(out,dim=1)
 
class MyTestDataset(torch.utils.data.Dataset):
	def __init__(self,path2, datacsv, transform=None, target_transform=None):
		super(MyTestDataset,self).__init__()
		names = []
		with open(datacsv,'r') as file:
			reader=csv.reader(file)
			for line in reader:
				names.append(line[0])
		self.names = names
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		name= self.names[index]
		with np.load(os.path.join(path2, '%s.npz' % name)) as npz:
			tmp_voxel = npz['voxel']
			
			voxel = tmp_voxel[34:66,34:66,34:66]
			if self.transform is not None:
				voxel = self.transform(voxel) 
			voxel=np.expand_dims(voxel,axis=0)
			voxel=torch.from_numpy(voxel)
		return voxel,name

	def __len__(self):
		return len(self.names)

test_data=MyTestDataset(path2,test_file_name,transform=transform)
test_loader=data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model=torch.load('learn01.tar')
if torch.cuda.is_available():
	model=model.cuda()
model.eval()


f = open('submission.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(["Id","Predicted"])
for voxel,name in tqdm(test_loader):
	if torch.cuda.is_available():
		voxel=voxel.cuda()
	out=model(voxel)
	ndarray = out.cpu().detach().numpy()
	for i in range(batch_size):
		csv_writer.writerow([name[i],ndarray[i][1]])
	print(name,out)