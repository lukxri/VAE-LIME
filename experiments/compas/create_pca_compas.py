import torch
from sklearn.preprocessing import MinMaxScaler

from get_data import *
from train_compas_vae import VAE

try:
    params = Params("../../fooling_lime/model_configurations/experiment_params.json")
except FileNotFoundError:
    params = Params("fooling_lime/model_configurations/experiment_params.json")

np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data(params)
# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0, 1], size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0, 1], size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')

X = X.values
categorical = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'),
          features.index('race'), features.index("sex_Male"), features.index("sex_Female")]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

numerical_only = False
if numerical_only:
    X = np.delete(X, categorical, axis=1)

    model = VAE(X.shape[1]).to(device)
    model.load_state_dict(torch.load("./vae_lime_compas_only_numerical.pt"))
    model.eval()

else:
    model = VAE(11).to(device)
    model.load_state_dict(torch.load("./vae_lime_compas.pt"))
    model.eval()

ss = MinMaxScaler().fit(X)
X = ss.transform(X)

print(X.shape)
#print(np.asarray(X).dtype)
num_samples = X.shape[0]
num_cols = X.shape[1]
"""
r = []
with torch.no_grad():
    # print("___________________________________________")
    # print("Generating 5 new data points using the VAE:\n")
    sample = torch.randn(num_samples, 30).to(device)

    # TODO Idea: Encode data row once, and sample from generated latent space.
    sample = model.decode(sample).cpu()
    data = sample.cpu().numpy().reshape(-1, num_cols)

    data[:, categorical] = [np.round(i, 0) for i in data[:, categorical]]
    # data = ss.inverse_transform(data)
    X_p = data
    r.append(X_p)"""

r = []
for _ in range(1):
    p = np.random.normal(0, 1, size=X.shape)

    # for row in p:
    # 	for c in c_cols:
    # 		row[c] = np.random.choice(X[:,c])

    X_p = X + p
    r.append(X_p)

r = np.vstack(r)
p = [1 for _ in range(len(r))]
iid = [0 for _ in range(len(X))]

all_x = np.vstack((r, X))
all_y = np.array(p + iid)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
results = pca.fit_transform(all_x)

print(len(X))

plt.scatter(results[:500, 0], results[:500, 1], alpha=.3, c="r", label="VAE generated data")
plt.scatter(results[-500:, 0], results[-500:, 1], alpha=.3, c="b", label="Original data")
plt.legend()

if numerical_only:
    plt.title("VAE-LIME COMPAS data | Numerical features")
else:
    plt.title("VAE-LIME COMPAS data | All features")
plt.show()
