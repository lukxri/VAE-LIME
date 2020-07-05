import torch
from sklearn.preprocessing import MinMaxScaler

from get_data import *
from train_german_vae import VAE

try:
    params = Params("model_configurations/experiment_params.json")
except FileNotFoundError:
    params = Params("../../fooling_lime/model_configurations/experiment_params.json")

X, y, cols = get_and_preprocess_german(params)
features = [c for c in X]

X = X.values
categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
               'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
               'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
               'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
               'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

numerical_only = True
if numerical_only:
    X = np.delete(X, categorical, axis=1)

    model = VAE(X.shape[1]).to(device)
    model.load_state_dict(torch.load("./vae_lime_german_only_numerical_test2.pt"))
    model.eval()

else:
    model = VAE(X.shape[1]).to(device)
    model.load_state_dict(torch.load("./vae_lime_german_test2.pt"))
    model.eval()

ss = MinMaxScaler().fit(X)
X = ss.transform(X)

print(X.shape)
print(np.asarray(X).dtype)
num_samples = X.shape[0]
num_cols = X.shape[1]

r = []
with torch.no_grad():
    # print("___________________________________________")
    # print("Generating 5 new data points using the VAE:\n")
    sample = torch.randn(num_samples, 30).to(device)
    sample = model.decode(sample).cpu()

    X = np.asarray(X, dtype=np.float32)
    data = sample.cpu().numpy().reshape(-1, num_cols)

    # data[:, categorical] = [np.round(i, 0) for i in data[:, categorical]]
    # data = ss.inverse_transform(data)
    X_p = data
    r.append(X_p)

# old perturbation sampling
"""r = []
for _ in range(1):
    p = np.random.normal(0, 1, size=X.shape)

    # for row in p:
    # 	for c in c_cols:
    # 		row[c] = np.random.choice(X[:,c])

    X_p = X + p
    r.append(X_p)"""

r = np.vstack(r)
p = [1 for _ in range(len(r))]
iid = [0 for _ in range(len(X))]

all_x = np.vstack((r, X))
all_y = np.array(p + iid)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=1)
results = pca.fit_transform(all_x)

print(len(X))

plt.scatter(results[:500, 0], results[:500, 1], alpha=.3, c="r", label="VAE generated data")
plt.scatter(results[-500:, 0], results[-500:, 1], alpha=.3, c="b", label="Original data")
plt.legend()
plt.xlim(-4, 4)
plt.ylim(-4, 4)

if numerical_only:
    plt.title("VAE-LIME German data | Numerical features")
else:
    plt.title("VAE-LIME German data | All features")
plt.show()
