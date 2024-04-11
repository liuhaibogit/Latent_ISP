import os
from os.path import join
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, LongTensor
from torch_geometric.nn import SplineConv, InstanceNorm, TopKPooling, ChebConv
import utils
import plotly.graph_objects as go
import torch_scatter as ts


class splineCNN(torch.nn.Module):
    def __init__(self):

        super(splineCNN, self).__init__()
        self.fc1 = nn.Linear(12, 1)

        self.fc2 = nn.Linear(7, 16)

        self.conv1 = SplineConv(16, 64, dim=1, degree=1, kernel_size=20)
        self.conv2 = SplineConv(64, 128, dim=1, degree=1, kernel_size=20)
        self.conv3 = SplineConv(128, 64, dim=1, degree=1, kernel_size=20)
        self.conv4 = SplineConv(64, 32, dim=1, degree=1, kernel_size=20)
        self.conv5 = SplineConv(32, 16, dim=1, degree=1, kernel_size=20)
        self.conv6 = SplineConv(16, 1, dim=1, degree=1, kernel_size=20)


        self.dropout = nn.Dropout(p = 0.2)



    def forward(self, verts, edges, faces, normals):



        # total_area
        f0 = faces
        f1 = torch.ones(faces.shape[0], 3).long().cuda()
        f1[:, 0] = faces[:, 1]
        f1[:, 1] = faces[:, 0]
        f1[:, 2] = faces[:, 2]
        f2 = torch.ones(faces.shape[0], 3).long().cuda()
        f2[:, 0] = faces[:, 2]
        f2[:, 1] = faces[:, 0]
        f2[:, 2] = faces[:, 1]
        faces_perm = torch.cat((f0, f1, f2), dim=0)
        tangent1 = verts[faces_perm[:, 1]] - verts[faces_perm[:, 0]]
        tangent2 = verts[faces_perm[:, 2]] - verts[faces_perm[:, 0]]
        area_per_face = tangent1.cross(tangent2).norm(dim=1) / 2
        total_area = ts.scatter_add(area_per_face, faces_perm[:, 0], dim=0,
                                    out=torch.zeros(verts.shape[0]).cuda())


        # edge_attr = self.fc1(torch.cat([verts[edges[0, :], :], verts[edges[1, :], :], normals[edges[0, :], :], normals[edges[1, :], :]], dim=1))

        edge_attr = torch.norm(verts[edges[0, :], :] - verts[edges[1, :], :], dim=1).unsqueeze(1)

        x = self.fc2(torch.cat([verts, total_area.unsqueeze(1), normals], dim=1))

        x = F.elu(self.conv1(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv2(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv3(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv4(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv5(x, edges, edge_attr))
        x = self.dropout(x)

        out = self.conv6(x, edges, edge_attr)

        return out.squeeze()



class splineCNN_copy(torch.nn.Module):
    def __init__(self):

        super(splineCNN_copy, self).__init__()
        self.fc1 = nn.Linear(6, 1)

        self.fc2 = nn.Linear(4, 16)

        self.conv1 = SplineConv(16, 32, dim=1, degree=1, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=1, degree=1, kernel_size=10)
        self.conv3 = SplineConv(64, 128, dim=1, degree=1, kernel_size=20)
        self.conv4 = SplineConv(128, 64, dim=1, degree=1, kernel_size=20)
        self.conv5 = SplineConv(64, 32, dim=1, degree=1, kernel_size=10)
        self.conv6 = SplineConv(32, 16, dim=1, degree=1, kernel_size=5)

        self.fc3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p = 0.2)  # dropout训练



    def forward(self, verts, edges, faces, normals):
        # total_area
        f0 = faces
        f1 = torch.ones(faces.shape[0], 3).long().cuda()
        f1[:, 0] = faces[:, 1]
        f1[:, 1] = faces[:, 0]
        f1[:, 2] = faces[:, 2]
        f2 = torch.ones(faces.shape[0], 3).long().cuda()
        f2[:, 0] = faces[:, 2]
        f2[:, 1] = faces[:, 0]
        f2[:, 2] = faces[:, 1]
        faces_perm = torch.cat((f0, f1, f2), dim=0)
        tangent1 = verts[faces_perm[:, 1]] - verts[faces_perm[:, 0]]
        tangent2 = verts[faces_perm[:, 2]] - verts[faces_perm[:, 0]]
        area_per_face = tangent1.cross(tangent2).norm(dim=1) / 2
        total_area = ts.scatter_add(area_per_face, faces_perm[:, 0], dim=0,
                                    out=torch.zeros(verts.shape[0]).cuda())

        edge_attr = torch.norm(verts[edges[0, :], :] - verts[edges[1, :], :], dim=1).unsqueeze(1)

        # edge_attr = self.fc1(torch.cat([verts[edges[0, :], :], verts[edges[1, :], :]], dim=1))

        x = self.fc2(torch.cat([verts, total_area.unsqueeze(1)], dim=1))

        x = F.elu(self.conv1(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv2(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv3(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv4(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv5(x, edges, edge_attr))
        x = self.dropout(x)

        x = F.elu(self.conv6(x, edges, edge_attr))
        x = self.dropout(x)

        out = self.fc3(x)

        return out.squeeze()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = join(os.path.dirname(os.path.realpath(__file__)), '..')
    train_dataset = utils.heldata(root + '/data/helmholtz/solution', train=True)
    train_loader = torch_geometric.data.DataLoader(train_dataset,
                                                   shuffle=True)
    model = splineCNN().to(device)
    print(utils.num_parameters(model))

    v, e, f, n, partial, index, name = train_dataset.__getitem__(1)
    v = v.to(device)
    e = e.to(device)
    f = f.to(device)
    n = n.to(device)
    out = model(v, e, f, n)
    fig = go.Figure(data=[
        go.Scatter3d(
            x=v.cpu().detach().numpy()[:, 0],
            y=v.cpu().detach().numpy()[:, 1],
            z=v.cpu().detach().numpy()[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=out.cpu().detach().numpy(),  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8,
                colorbar=dict(thickness=50)
            )
        )
    ])
    fig.show()


