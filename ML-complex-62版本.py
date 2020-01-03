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

input_file_name='train_val.csv'
#训练集和验证集的name label信息
test_file_name='sampleSubmission.csv'
#借助处理后的sampleSubmission.scv文件读取测试集name
path="train_val"
#训练集和验证集所在文件夹
path2="test"
#测试集所在文件夹
learning_rate=1e-3
NUM_EPOCHS=50
is_mixup = True
batch_size=64
rot_flag=True

# preprocessing


normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

class MyDataset(torch.utils.data.Dataset):
	def __init__(self,path, datacsv,flag, transform=None, target_transform=None):
		super(MyDataset,self).__init__()
		imgs = []
		with open(datacsv,'r') as file:
			reader=csv.reader(file)
			for line in reader:
				imgs.append((line[0],int(line[1])))
		self.flag=flag
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		if self.flag==0:
			name, label = self.imgs[index]
		if self.flag==1:
			name, label = self.imgs[index]
		with np.load(os.path.join(path, '%s.npz' % name)) as npz:
			tmp_voxel = npz['voxel']
			z = np.random.randint(0, 2)
			x = np.random.randint(0, 2)
			y = np.random.randint(0, 2)
			if (z == 1):
				tmp_voxel = np.flip(tmp_voxel, 0)
			if (x == 1):
				tmp_voxel = np.flip(tmp_voxel, 1)
			if (y == 1):
				tmp_voxel = np.flip(tmp_voxel, 2)
			tmp_voxel= tmp_voxel.copy()
			voxel = tmp_voxel[34:66,34:66,34:66]
			
			if self.transform is not None:
				voxel = self.transform(voxel)
			voxel=np.expand_dims(voxel,axis=0)
			voxel=torch.from_numpy(voxel)

		return voxel, label

	def __len__(self):

		return len(self.imgs)

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
def mixup_data(x, y, alpha=1.0, use_cuda=True):
#'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
	
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
 
#data定义
full_dataset=MyDataset(path,input_file_name,flag=0,transform=transform)

dataset_size = len(full_dataset)
#train和val以4:1划分
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
from torch.utils.data.sampler import SubsetRandomSampler
train_sampler=SubsetRandomSampler(train_indices)
val_sampler =SubsetRandomSampler(val_indices)

test_data=MyTestDataset(path2,test_file_name,transform=transform)


train_loader =data.DataLoader(dataset=full_dataset, batch_size=batch_size,sampler=train_sampler,drop_last=True)
val_loader =data.DataLoader(dataset=full_dataset, batch_size=batch_size,sampler=val_sampler)

test_loader=data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model=xnet()
if torch.cuda.is_available():
	model=model.cuda()
#define loss function and optimiter
criterion =nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr =learning_rate, momentum=0.9)

train_loss=[]
val_loss=[]
val_acc=[]
for epoch in range(NUM_EPOCHS):
	running_loss=0.0

	if epoch > 100:
		is_mixup = False
	else:
		is_mixup = True

	model.train()
	for voxel, label in tqdm(train_loader):
#forward + backward + optimize

		if is_mixup:
			inputs, targets_a, targets_b, lam = mixup_data(voxel.cuda(), label.cuda(), alpha=1.0)
			inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
			outputs = model(inputs)
			optimizer.zero_grad()
			loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
			loss.backward()
			optimizer.step()
		else:
			if torch.cuda.is_available():
				voxel=voxel.cuda()
				label=label.cuda()
			else:
				voxel=Variable(voxel)
				label=Variable(label)
			
			out=model(voxel)
			loss=criterion(out,label)
			print_loss=loss.data.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		running_loss+=loss.item()
		if epoch%10==0:
			print('epoch:{},loss:{:.4f}'.format(epoch,loss.data.item()))
	print ('running_loss:',running_loss)
	train_loss.append(running_loss/len(train_loader))
	
	model.eval()
	eval_loss=0
	eval_acc=0

	for voxel, label in tqdm(val_loader):

		if torch.cuda.is_available():
			voxel=voxel.cuda()
			label=label.cuda()

		out=model(voxel)
		loss=criterion(out,label)
		eval_loss+=loss.item()*label.size(0)
		_,pred=torch.max(out,1)
		num_correct=(pred==label).sum()
		eval_acc+=num_correct.item()
	print('eval_loss:',eval_loss)
	val_loss.append(eval_loss/len(val_loader))
	print('eval_accuracy',(eval_acc/split))
	val_acc.append(eval_acc/split)
	
torch.save(model, 'learn01.tar')
print('train_loss:',train_loss)
print('val_loss:',val_loss)
print('val_acc:',val_acc)

model.eval()
eval_loss=0
eval_acc=0

for voxel, label in tqdm(val_loader):

	if torch.cuda.is_available():
		voxel=voxel.cuda()
		label=label.cuda()

	out=model(voxel)
	loss=criterion(out,label)
	eval_loss+=loss.item()*label.size(0)
	_,pred=torch.max(out,1)
	num_correct=(pred==label).sum()
	eval_acc+=num_correct.item()
	print(eval_loss)




val_accuracy=eval_acc/split
print('Val accuracy: %0.2f%%' % (val_accuracy*100))

eval_loss=0
eval_acc=0
for voxel, label in tqdm(train_loader):

	if torch.cuda.is_available():
		voxel=voxel.cuda()
		label=label.cuda()

	out=model(voxel)
	loss=criterion(out,label)
	eval_loss+=loss.item()*label.size(0)
	_,pred=torch.max(out,1)
	num_correct=(pred==label).sum()
	eval_acc+=num_correct.item()
val_accuracy=eval_acc/(dataset_size-split)
print('Train accuracy: %0.2f%%' % (val_accuracy*100))

f = open('Submission.csv','w',encoding='utf-8',newline='')
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

x = np.arange(0,NUM_EPOCHS)
plt.plot(x, train_loss, color="r", linestyle="-", marker="^", linewidth=1)

plt.plot(x,val_loss, color="b", linestyle="-", marker="s", linewidth=1)

plt.xlabel("x")
plt.ylabel("y")
plt.show()
