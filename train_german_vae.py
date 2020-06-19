from __future__ import print_function

import warnings

warnings.filterwarnings('ignore')

import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fooling_lime import get_data, utils

# Set up experiment parameters
params = utils.Params("fooling_lime/model_configurations/experiment_params.json")
X, y, cols = get_data.get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

train_only_numerical = True
if train_only_numerical:
    categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
                   'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
                   'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
                   'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
                   'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']
    categorical = [features.index(c) for c in categorical]
    X = np.delete(X, categorical, axis=1)
    num_cols = 7
else:
    num_cols = 28

# TODO Fit different scalers to categorial and numerical data
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

#scaler = MinMaxScaler().fit(xtrain)
#xtrain = scaler.transform(xtrain)
#xtest = scaler.transform(xtest)

# mean_lrpi = np.mean(xtrain[:, loan_rate_indc])

parser = argparse.ArgumentParser(description='VAE LIME')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#xtrain = np.delete(xtrain, [3, 4, 5, 6, 7, 8, 9], axis=1)
# ytrain = np.delete(ytrain, [3, 4, 5, 6, 7, 8, 9])
#xtest = np.delete(xtest, [3, 4, 5, 6, 7, 8, 9], axis=1)
# ytest = np.delete(ytest, [3, 4, 5, 6, 7, 8, 9])

# Prepare data loader
xtrain = torch.from_numpy(xtrain)
ytrain = torch.from_numpy(ytrain)
xtest = torch.from_numpy(xtest)
ytest = torch.from_numpy(ytest)

train_dataset = TensorDataset(xtrain, ytrain)
test_dataset = TensorDataset(xtest, ytest)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, num_cols):
        super(VAE, self).__init__()

        self.num_cols = num_cols
        # TODO neural net hyper-parameters
        """self.fc1 = nn.Linear(28, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28)"""

        self.fc1 = nn.Linear(self.num_cols, 60)
        self.fc21 = nn.Linear(60, 30)
        self.fc22 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 60)
        self.fc4 = nn.Linear(60, self.num_cols)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.num_cols))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(num_cols).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            #print("Mu: {} \t logvar: {}".format(mu.shape, logvar.shape))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    print("Mu: {} \t logvar: {}".format(mu.shape, logvar.shape))
    #print("Mu: {} \t logvar: {}".format(mu[0].mean(), logvar[0].mean()))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            """
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            """
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

        if args.save_model:
            if train_only_numerical:
                torch.save(model.state_dict(), "vae_lime_german_only_numerical.pt")
            else:
                torch.save(model.state_dict(), "vae_lime_german.pt")



    with torch.no_grad():
        print("___________________________________________")
        print("Generating 5 new data points using the VAE:\n")
        sample = torch.randn(10, 30).to(device)
        sample = model.decode(sample).cpu()

        # TODO Inverse transform not one-hot ?!
        inversed = scaler.inverse_transform(sample)
        np.set_printoptions(suppress=True)
        #print(sample)
        #print(inversed)

        s = [np.round(i, 0) for i in inversed]
        for a in s:
            print(a)

        # TODO Test how unique the samples truly are


if __name__ == "__main__":
    main()
