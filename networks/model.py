import os
from os.path import join
import torch
import torch_geometric
from torch import nn
import utils
import torch_geometric as pyg
from torch_geometric.nn import Linear as Linear_pyg
import plotly.graph_objects as go
import torch_scatter as ts



class GATCNN(torch.nn.Module):
    def __init__(self):

        super(GATCNN, self).__init__()

        self.fc1 = nn.Linear(7, 128)

        gin_nn = nn.Sequential(
            Linear_pyg(128, 256), nn.ReLU(),
            Linear_pyg(256, 256))
        self.conv2 = pyg.nn.GINConv(gin_nn)
        gin_nn = nn.Sequential(
            Linear_pyg(256, 512), nn.ReLU(),
            Linear_pyg(512, 512))
        self.conv3 = pyg.nn.GINConv(gin_nn)
        gin_nn = nn.Sequential(
            Linear_pyg(512, 256), nn.ReLU(),
            Linear_pyg(256, 256))
        self.conv5 = pyg.nn.GINConv(gin_nn)
        gin_nn = nn.Sequential(
            Linear_pyg(256, 128), nn.ReLU(),
            Linear_pyg(128, 128))
        self.conv6 = pyg.nn.GINConv(gin_nn)

        self.fc3 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(p = 0.2)



    def forward(self, vertices, edge_index, faces, total_area, normals):

        total_area = total_area.unsqueeze(1)

        x = self.fc1(torch.cat([vertices, total_area, normals], dim=1))

        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()

        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()

        x = self.dropout(x)
        x = self.conv5(x, edge_index).relu()

        x = self.dropout(x)
        x = self.conv6(x, edge_index)

        out = self.fc3(x).squeeze()

        return out




if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = join(os.path.dirname(os.path.realpath(__file__)), '..')
    train_dataset = utils.heldata(root + '/data/helmholtz/solution', train=True)
    train_loader = torch_geometric.data.DataLoader(train_dataset,
                                                   shuffle=True)
    model = GATCNN().to(device)
    print(utils.num_parameters(model))
    v, e, f, t, n, partial, index, name = train_dataset.__getitem__(1)
    v = v.to(device)
    e = e.to(device)
    f = f.to(device)
    t = t.to(device)
    n = n.to(device)
    out = model(v, e, f, t, n)
    fig = go.Figure(data=[
        go.Scatter3d(
            x=v.cpu().detach().numpy()[:, 0],
            y=v.cpu().detach().numpy()[:, 1],
            z=v.cpu().detach().numpy()[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=out.cpu().detach().numpy(),
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(thickness=50)
            )
        )
    ])
    fig.show()




