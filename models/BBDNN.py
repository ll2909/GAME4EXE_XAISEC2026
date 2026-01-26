import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BBDNN(nn.Module):
    def __init__(self, min_len = 4096, max_len = 102400):
        super(BBDNN, self).__init__()
        self.min_len = min_len
        self.max_len = max_len

        self.embedding_1 = torch.nn.Embedding(
            num_embeddings=257, embedding_dim=10, padding_idx=256
        )
        self.conv1d_1 = torch.nn.Conv1d(
            in_channels=10,
            out_channels=96,
            kernel_size=(11,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_2 = torch.nn.Conv1d(
            in_channels=96,
            out_channels=128,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_3 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_4 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_5 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.dense_1 = torch.nn.Linear(in_features=512, out_features=1, bias=True)

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
        conv1d_1_activation = self.relu(conv1d_1)
        max_pooling1d_1 = torch.max_pool1d(
            conv1d_1_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_2 = self.conv1d_2(max_pooling1d_1)
        conv1d_2_activation = self.relu(conv1d_2)
        max_pooling1d_2 = torch.max_pool1d(
            conv1d_2_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_3 = self.conv1d_3(max_pooling1d_2)
        conv1d_3_activation = self.relu(conv1d_3)
        max_pooling1d_3 = torch.max_pool1d(
            conv1d_3_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_4 = self.conv1d_4(max_pooling1d_3)
        conv1d_4_activation = self.relu(conv1d_4)
        max_pooling1d_4 = torch.max_pool1d(
            conv1d_4_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_5 = self.conv1d_5(max_pooling1d_4)
        conv1d_5_activation = self.relu(conv1d_5)
        global_max_pooling1d_1 = torch.max_pool1d(
            input=conv1d_5_activation, kernel_size=conv1d_5_activation.size()[2:]
        )
        global_average_pooling1d_1 = torch.avg_pool1d(
            input=conv1d_5_activation, kernel_size=conv1d_5_activation.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )
        global_average_pooling1d_1_flatten = global_average_pooling1d_1.view(
            global_average_pooling1d_1.size(0), -1
        )
        concatenate_1 = torch.cat(
            (global_max_pooling1d_1_flatten, global_average_pooling1d_1_flatten), 1
        )
        dense_1 = self.dense_1(concatenate_1)
        
        out = self.sigmoid(dense_1)
        return out
    

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved in ", path)


def load_model(path):
    model = BBDNN()
    model.load_state_dict(torch.load(path, map_location=device))
    print("Model loaded")
    return model