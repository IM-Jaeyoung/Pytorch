from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

"""Training Settings"""
parser = argparse.ArgumentParser(description="Pytorch MNIST_VAE Example")
parser.add_argument('--batch-size', type=int, default=16, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N')
parser.add_argument('--epochs', type=int, default=10000, metavar='N')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


"""Data Loader"""
data_transform = transforms.Compose([
	transforms.Resize((256, 1024)),
	transforms.ToTensor()])

test_kitti_dataset= datasets.ImageFolder(root='/home/jyim/hdd2/pytorch_KITTI/test_for_save', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_kitti_dataset, batch_size=args.test_batch_size, shuffle=None, num_workers=1)
"""Define Networks"""
### Variational Auto-Encoder
class VAE_KITTI(nn.Module):
	### Kitti size = [3, 256, 1024] (resized image)
	def __init__(self):
		super (VAE_KITTI, self).__init__()
		### Encoding parts
		self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

		self.fc1 = nn.Linear(8192, 2048)
		self.fc2 = nn.Linear(2048, 512)
		
		self.fc_mu = nn.Linear(512, 64)
		self.fc_logvar = nn.Linear(512, 64)

		### Decoding parts
		self.up_fc1 = nn.Linear(64, 512)
		self.up_fc2 = nn.Linear(512, 2048)
		self.up_fc3 = nn.Linear(2048, 8192)

		self.up_conv1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
		self.up_conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
		self.up_conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
		self.up_conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
		self.up_conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
		self.up_conv6 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
		self.up_conv7 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
		### Activation parts
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def encoder(self, x):
		h1 = self.relu(F.max_pool2d(self.conv1(x), 2))	# [B,3,256,1024] -> [B,32,128,512]
		h2 = self.relu(F.max_pool2d(self.conv2(h1), 2))	# [B,32,128,512] -> [B,64,64,256]
		h3 = self.relu(F.max_pool2d(self.conv3(h2), 2))	# [B,64,64,256] -> [B,128,32,128]
		h4 = self.relu(F.max_pool2d(self.conv4(h3), 2))	# [B,128,32,128] -> [B,256,16,64]
		h5 = self.relu(F.max_pool2d(self.conv5(h4), 2))	# [B,256,16,64] -> [B,512,8,32]
		h6 = self.relu(F.max_pool2d(self.conv6(h5), 2))	# [B,512,8,32] -> [B,512,4,16]
		h7 = self.relu(F.max_pool2d(self.conv7(h6), 2))	# [B,512,4,16] -> [B,512,2,8]
		h7_vec = h7.view(-1, 8192)						# [B,512,2,8] -> [B,8192]

		h8 = self.relu(self.fc1(h7_vec))					# [B,8192] -> [B,2048]
		h9 = self.relu(self.fc2(h8))					# [B,2048] -> [B,512]

		z_mu = self.fc_mu(h9)							# [B,512] -> [B,64]
		z_logvar = self.fc_logvar(h9)					# [B,512] -> [B,64]
		return z_mu, z_logvar

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		if args.cuda:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)	#epsilon
		z = eps.mul(std).add_(mu)	# mean + sigma * epsilon [B,20]
		return z

	def decoder(self, z):
		h1 = self.relu(self.up_fc1(z))			# [B,64] -> [B,512]
		h2 = self.relu(self.up_fc2(h1))			# [B,512] -> [B,2048]
		h3 = self.relu(self.up_fc3(h2))			# [B,2048] -> [B,8192]
		h3_img = h3.view(-1, 512, 2, 8)			# [B,8192] -> [B,512,2,8]
		h4 = self.relu(self.up_conv1(h3_img))	# [B,512,2,8] -> [B,512,4,16]
		h5 = self.relu(self.up_conv2(h4))		# [B,512,4,16] -> [B,512,8,32]
		h6 = self.relu(self.up_conv3(h5))		# [B,512,8,32] -> [B,256,16,64]
		h7 = self.relu(self.up_conv4(h6))		# [B,256,16,64] -> [B,128,32,128]
		h8 = self.relu(self.up_conv5(h7))		# [B,128,32,128] -> [B,64,64,256]
		h9 = self.relu(self.up_conv6(h8))		# [B,64,64,256] -> [B,32,128,512]
		x_ = self.sigmoid(self.up_conv7(h9))	# [B,32,128,512] -> [B,3,256,1024]
		return x_

	def forward(self, x):
		z_mu, z_logvar = self.encoder(x)	# Encoding
		z = self.reparameterize(z_mu, z_logvar)	# Latent Variable
		x_ = self.decoder(z)	# Decoding
		return x_, z, z_mu, z_logvar

Net_VAE = VAE_KITTI()
if args.cuda:
	Net_VAE.cuda()

ckpoint = torch.load('/home/jyim/hdd2/VAE_KITTI/model_700.ckpt')
Net_VAE.load_state_dict(ckpoint['VAE_state_dict'])

"""Define Test"""
def test():
    Net_VAE.eval()
    for batch_idx, (data, _) in enumerate(test_loader):
        ### Prepare Input Data, Affine Transformed Data, GT
        x = Variable(data)
        if args.cuda:
        	x = x.cuda()
       
        # Inference
        re_x, z, mu, logvar = Net_VAE(x)

        for batch_num in range(args.test_batch_size):
        	name = '/home/jyim/hdd2/VAE_KITTI/result_imgs/batch_' + str(batch_idx) + '_idx_' + str(batch_num) + '.png'
        	torchvision.utils.save_image(re_x[batch_num,:,:,:], name)


test()