from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import skimage.measure
from utils.sdf import _cartesian_product



def indicator_plane(
        decoder, latent, N=48, l=0.05
):
    device = latent.device
    num_samp_per_scene = N ** 3
    bound = ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2))  # bounds
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P).float()

    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int().to(device), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(torch.cat([batch_vecs, xyz], dim=1).to(device)).detach().cpu()

    return torch.sign(input-l)



def create_mesh_with_edge(decoder, latent, N=32, l=0.03):
    num_samp_per_scene = N ** 3
    bound = ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2))
    # bound = ((-1, -1, -1), (1, 1, 1))
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P).float()
    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int(), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(torch.cat([batch_vecs, xyz], dim=1))
    numpy_3d_sdf_tensor = input.squeeze().detach().cpu().numpy().reshape(N, N, N)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=l, spacing=[N] * 3
    )

    # verts_scaled = verts / (N ** 2) * 2 - 1

    # 计算每个体素的实际物理尺寸（spacing）
    spacing = [(x1 - x0) / (N - 1)**2, (y1 - y0) / (N - 1)**2, (z1 - z0) / (N - 1)**2]
    # 假设 verts 已经通过 marching_cubes 生成
    # 调整 verts 到真实的空间坐标系
    verts_scaled = np.zeros_like(verts)
    verts_scaled[:, 0] = verts[:, 0] * spacing[0] + x0
    verts_scaled[:, 1] = verts[:, 1] * spacing[1] + y0
    verts_scaled[:, 2] = verts[:, 2] * spacing[2] + z0

    # verts_scaled 现在包含归一化调整后的顶点空间坐标

    return verts_scaled, faces, normals



def write_verts_faces_fields_to_file(verts, faces, field, ply_filename_out):
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            colorbar_title='',
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # colorscale='viridis',
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = field,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name='y',
            showscale=True
        )
    ])
    # 设置坐标轴比例不变
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )
    fig.write_html(ply_filename_out)

    plt.close()



def write_verts_faces_to_file(verts, faces, ply_filename_out):
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name='y',
            showscale=True
        )
    ])
    # 设置坐标轴比例不变
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )
    fig.write_html(ply_filename_out)