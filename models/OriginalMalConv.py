import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OriginalMalConv(nn.Module):
    def __init__(self, max_len = 2**20):
        super(OriginalMalConv, self).__init__()
        self.max_len = max_len

        self.embedding_1 = nn.Embedding(num_embeddings=257, embedding_dim=8, padding_idx=256)
        self.conv1d_1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=(500,),
                                  stride=(500,),
                                  groups=1, bias=True)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=(500,),
                                  stride=(500,),
                                  groups=1, bias=True)
        self.dense_1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.dense_2 = nn.Linear(in_features=128, out_features=1, bias=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def embed(self, x):
        emb_x = self.embedding_1(x)
        return emb_x
    

    def forward(self, x, is_embedded = True):
        if not is_embedded:
            x = self.embed(x)

        x = x.transpose(-1, -2)
        conv1d_1 = self.conv1d_1(x)
        conv1d_2 = self.conv1d_2(x)
        conv1d_1_activation = self.relu(conv1d_1)
        conv1d_2_activation = self.sigmoid(conv1d_2)
        multiply_1 = conv1d_1_activation * conv1d_2_activation
        global_max_pooling1d_1 = F.max_pool1d(
            input=multiply_1, kernel_size=multiply_1.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )
        dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
        dense_1_activation = self.relu(dense_1)
        dense_2 = self.dense_2(dense_1_activation)

        out = self.sigmoid(dense_2)
        return out


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved in ", path)


def load_model(path):
    model = OriginalMalConv()
    model.load_state_dict(torch.load(path, map_location=device))
    print("Model loaded")
    return model