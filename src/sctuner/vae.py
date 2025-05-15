from tqdm import tqdm
import torch
import torch.nn as nn
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sctuner.optimisers import AdEMAMix

class VAE(nn.Module):

    def __init__(self, input_dim=2000, hidden_dim=128, latent_dim=20, meanlog_dim=8,device=device): #400 & 200 first
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, meanlog_dim) # 8 worked well
        self.logvar_layer = nn.Linear(latent_dim, meanlog_dim) # 8 worked well
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(meanlog_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=0) # first Sigmoid
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# Define parameters for def train
model = VAE().to(device)
optimizer = AdEMAMix(model.parameters())

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss , KLD


def train(model, optimizer, epochs, device, train_loader = None):
    model.train()
    overall_loss = 0

    pbar = tqdm(range(epochs), desc="Epochs")
    for _ in pbar:

        for batch in train_loader:
            data = batch.to(device, non_blocking=True) #normally this one as well: target = target.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(data)
            loss, KLD = loss_function(data, x_hat, mean, log_var)
            loss_KLD = loss + KLD
            overall_loss += loss_KLD.item()
            
            loss.backward()
            optimizer.step()

        train_loss = overall_loss/len(train_loader.dataset)
        i = int(train_loss)
        pbar.set_postfix({"Current loss VAE": i})
        pass
    return overall_loss