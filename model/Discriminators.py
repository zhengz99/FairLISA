import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.criterion = nn.BCELoss()

        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, args.LATENT_NUM),
            nn.ReLU(),
            nn.Linear(args.LATENT_NUM, args.LATENT_NUM),
            nn.ReLU(),
            nn.Linear(args.LATENT_NUM, 1),
        ).to("cuda")

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)
        loss = self.criterion(output, labels)
        return loss

    def hforward(self, embeddings):
        output = self.predict(embeddings)
        loss = torch.sum(
            -output * torch.log(output + 1e-8)
            - (1 - output) * torch.log(1 - output + 1e-8)
        ) / len(output)
        return loss

    def predict(self, embeddings):
        output = torch.squeeze(torch.sigmoid(self.network(embeddings)), dim=-1)
        return output

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
