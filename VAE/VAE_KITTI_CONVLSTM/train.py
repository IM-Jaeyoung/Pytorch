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

writer = SummaryWriter('/home/jyim/hdd2/VAE_KITTI/runs')

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

train_kitti_dataset= datasets.ImageFolder(root='/home/jyim/hdd2/pytorch_KITTI/train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_kitti_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

test_kitti_dataset= datasets.ImageFolder(root='/home/jyim/hdd2/pytorch_KITTI/test', transform=data_transform)
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

"""Define Loss"""
### Loss for VAE
def Loss_VAE(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

"""Define Optimizer"""
params = list(Net_VAE.parameters())
optimizer = optim.Adam(params, lr=args.lr)


"""Utils"""

"""Define Train"""
def train(epoch):
    Net_VAE.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        ### Prepare Input Data, Affine Transformed Data, GT
        x = Variable(data)
        if args.cuda:
        	x = x.cuda()
       
        optimizer.zero_grad()

        # Inference
        re_x, z, mu, logvar = Net_VAE(x)

        # Loss
        loss_VAE1 = Loss_VAE(re_x, x, mu, logvar)
        loss_total = loss_VAE1
        loss_total.backward()
        train_loss += loss_total.data
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_total.data / len(data)))
            niter = epoch*len(train_loader)+batch_idx

            # Tensorboard
            writer.add_scalar('Train/1_Loss_VAE1', loss_VAE1.data, niter)
            writer.add_scalar('Train/4_Total_Loss', loss_total.data, niter)
            grid_input_1 = torchvision.utils.make_grid(x, nrow=1)
            grid_output_1 = torchvision.utils.make_grid(re_x, nrow=1)
            writer.add_image('Train/1_Original', grid_input_1, niter)
            writer.add_image('Train/2_Reconstruction', grid_output_1, niter)
            

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

"""Define Test"""
def test(epoch):
    Net_VAE.eval()
    for batch_idx, (data, _) in enumerate(test_loader):
        ### Prepare Input Data, Affine Transformed Data, GT
        x = Variable(data)
        if args.cuda:
        	x = x.cuda()
       
        # Inference
        re_x, z, mu, logvar = Net_VAE(x)

        # Tensorboard
        # writer.add_scalar('Test/1_Loss_VAE1', loss_VAE1.data, niter)
        # writer.add_scalar('Test/4_Total_Loss', loss_total.data, niter)
        niter = epoch*len(test_loader)+batch_idx
        grid_input_1 = torchvision.utils.make_grid(x, nrow=1)
        grid_output_1 = torchvision.utils.make_grid(re_x, nrow=1)
        writer.add_image('Test/1_Original', grid_input_1, niter)
        writer.add_image('Test/2_Reconstruction', grid_output_1, niter)
        break




# writer.add_graph(model)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch % 10 == 0:
    	path = '/home/jyim/hdd2/VAE_KITTI/model_' + str(epoch) +'.ckpt'
    	torch.save({'VAE_state_dict': Net_VAE.state_dict()}, path)
    test(epoch)