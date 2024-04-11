from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import skimage.measure
import time
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge
# from torch_geometric.data import Data
# from torch_geometric.transforms import FaceToEdge
import utils
import torch_scatter as ts
from torch import LongTensor



# f = sphere(1.5)
# f.save('out.stl')
# import vtkplotlib as vpl
# from stl.mesh import Mesh
# mesh = Mesh.from_file("out.stl")
# fig = vpl.figure()
# mesh = vpl.mesh_plot(mesh)
# vpl.show()
from utils.sdf import _cartesian_product



def convert_sdf_samples_to_mesh(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        offset=None,
        scale=None,
):
    """
       Convert sdf samples to vertices,faces
       :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
       :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
       :voxel_size: float, the size of the voxels
       :ply_filename_out: string, path of the filename to save to
       This function is adapted from https://github.com/facebookresearch/DeepSDF
       """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.03, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces




def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    input = torch.cat([latent_repeat, queries], dim=1)
    sdf = decoder(input)
    return sdf




def create_mesh_encoder(
    input, r, filename = None
):
    level = torch.min(input) + 0.1 * (torch.max(input) - torch.min(input))
    numpy_3d_sdf_tensor = input.squeeze().detach().cpu().numpy().reshape(r, r, r)
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=level, spacing=[r] * 3
    )
    verts = verts / (r ** 2) * 2 - 1
    if filename==None:
        return verts, faces, normals, values
    else:
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name='',
                showscale=True
            )
        ])
        ply_filename_out = filename+'.html'
        fig.write_html(ply_filename_out)
        return




def create_mesh(
    decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None, output_mesh = False, filename = None
):
    start = time.time()
    ply_filename = filename


    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3]
        samples[head: min(head + max_batch, num_samples), 3] = (
            utils.decode_sdf(decoder, latent_vec, sample_subset)
                .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    if output_mesh is False:

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )
        return

    else:

        verts, faces = convert_sdf_samples_to_mesh(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            offset,
            scale,
        )

        # first fetch bins that are activated
        k = ((verts[:, 2] - voxel_origin[2]) / voxel_size).astype(int)
        j = ((verts[:, 1] - voxel_origin[1]) / voxel_size).astype(int)
        i = ((verts[:, 0] - voxel_origin[0]) / voxel_size).astype(int)
        # find points around
        next_samples = i * N * N + j * N + k
        next_samples_ip = np.minimum(i + 1, N - 1) * N * N + j * N + k
        next_samples_jp = i * N * N + np.minimum(j + 1, N - 1) * N + k
        next_samples_kp = i * N * N + j * N + np.minimum(k + 1, N - 1)
        next_samples_im = np.maximum(i - 1, 0) * N * N + j * N + k
        next_samples_jm = i * N * N + np.maximum(j - 1, 0) * N + k
        next_samples_km = i * N * N + j * N + np.maximum(k - 1, 0)

        next_indices = np.concatenate((next_samples, next_samples_ip, next_samples_jp, next_samples_kp, next_samples_im,
                                       next_samples_jm, next_samples_km))

        return verts, faces, samples, next_indices



def create_mesh_D(
    decoder, latent, N=48, l=0.03
):
    device = latent.device

    num_samp_per_scene = N ** 3
    bound = ((-1, -1, -1), (1, 1, 1))  # bounds
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P)
    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int().to(device), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(batch_vecs.to(device), xyz.to(device))
    numpy_3d_sdf_tensor = input.squeeze().detach().cpu().numpy().reshape(N, N, N)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=l, spacing=[N] * 3
    )
    verts = verts / (N ** 2) * 2 - 1

    return verts, faces, normals



def indicator_car(
        decoder, latent, N=48, l=0.05
):
    device = latent.device
    num_samp_per_scene = N ** 3
    bound = ((-1, -1, -1), (1, 1, 1))  # bounds
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P)
    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int().to(device), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(batch_vecs.to(device), xyz.to(device)).detach().cpu()

    return input




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
    xyz = torch.from_numpy(P)

    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int().to(device), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(torch.cat([batch_vecs, xyz], dim=1).to(device)).detach().cpu()

    return torch.sign(input-l)


def create_mesh_with_edge_L(decoder, latent, N=32, l=0.03):
    num_samp_per_scene = N ** 3
    bound = ((-1, -1, -1), (1, 1, 1))
    # bound = ((-0.6, -0.6, -0.6), (0.6, 0.6, 0.6))
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P)


    batch_vecs = torch.repeat_interleave(
        latent.unsqueeze(0), num_samp_per_scene * torch.ones(1, 1).squeeze().int(), dim=0)
    decoder.eval()
    with torch.no_grad():
        input = decoder(torch.cat([batch_vecs, xyz], dim=1))
    numpy_3d_sdf_tensor = input.squeeze().detach().cpu().numpy().reshape(N, N, N)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=l, spacing=[N] * 3
    )

    verts = verts / (N ** 2) * 2 - 1

    return verts, faces, normals



def create_mesh_with_edge(decoder, latent, N=32, l=0.03):
    num_samp_per_scene = N ** 3
    bound = ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2))
    # bound = ((-1, -1, -1), (1, 1, 1))
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, N)
    Y = np.linspace(y0, y1, N)
    Z = np.linspace(z0, z1, N)
    P = _cartesian_product(X, Y, Z)
    xyz = torch.from_numpy(P)
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





def convert_v_f_to_fig(
    v,f,name
):
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            name='',
            showscale=True
        )
    ])
    fig.write_html(name)



def convert_sdf_samples_to_fig(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf training to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is taken from https://github.com/facebookresearch/DeepSDF
    """
    start_time = time.time()


    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces,normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    verts = verts / (voxel_size ** 2) * 2 - 1

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
    fig.write_html(ply_filename_out)




def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf training to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is taken from https://github.com/facebookresearch/DeepSDF
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    write_verts_faces_to_file(verts, faces, ply_filename_out)


def write_EIT_to_file(verts, faces, voxel_size, partial, ply_filename_out):
    x, y, z, value = partial[:, 0], partial[:, 1], partial[:, 2], partial[:, 3]
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=12,
                color=value,    # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8,
                colorbar=dict(thickness=50)
            )
        ),
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
    fig.write_html(ply_filename_out)



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





def marching_cubes(
    decoder, latent_vec, level, N=64, max_batch=32 ** 3, offset=None, scale=None, output_mesh = False, filename = None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]
        samples[head : min(head + max_batch, num_samples), 3] = (
            utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    print(sdf_values.min(),sdf_values.max())



    if output_mesh is False:

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )
        return

    else:

        numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()

        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
        # transform from voxel coordinates to camera coordinates
        # note x and y are flipped in the output of marching_cubes
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        # apply additional offset and scale
        if scale is not None:
            mesh_points = mesh_points / scale
        if offset is not None:
            mesh_points = mesh_points - offset
        verts = mesh_points

        # first fetch bins that are activated
        k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
        j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
        i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
        # find points around
        next_samples = i*N*N + j*N + k
        next_samples_ip = np.minimum(i+1,N-1)*N*N + j*N + k
        next_samples_jp = i*N*N + np.minimum(j+1,N-1)*N + k
        next_samples_kp = i*N*N + j*N + np.minimum(k+1,N-1)
        next_samples_im = np.maximum(i-1,0)*N*N + j*N + k
        next_samples_jm = i*N*N + np.maximum(j-1,0)*N + k
        next_samples_km = i*N*N + j*N + np.maximum(k-1,0)

        next_indices = np.concatenate((next_samples,next_samples_ip, next_samples_jp,next_samples_kp,next_samples_im,next_samples_jm, next_samples_km))

        return verts, faces, normals, samples, next_indices








def create_mesh_optim_fast(
    samples, indices, decoder, latent_vec, N=128, max_batch=32 ** 3, offset=None, scale=None, fourier = False, taylor = False
):

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    num_samples = indices.shape[0]

    with torch.no_grad():

        head = 0
        while head < num_samples:
            sample_subset = samples[indices[head : min(head + max_batch, num_samples)], 0:3].reshape(-1, 3).cuda()
            samples[indices[head : min(head + max_batch, num_samples)], 3] = (
                utils.decode_sdf(decoder, latent_vec, sample_subset)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )


    # fetch bins that are activated
    k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
    j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
    i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
    # find points around
    next_samples = i*N*N + j*N + k
    next_samples_i_plus = np.minimum(i+1,N-1)*N*N + j*N + k
    next_samples_j_plus = i*N*N + np.minimum(j+1,N-1)*N + k
    next_samples_k_plus = i*N*N + j*N + np.minimum(k+1,N-1)
    next_samples_i_minus = np.maximum(i-1,0)*N*N + j*N + k
    next_samples_j_minus = i*N*N + np.maximum(j-1,0)*N + k
    next_samples_k_minus = i*N*N + j*N + np.maximum(k-1,0)
    next_indices = np.concatenate((next_samples,next_samples_i_plus, next_samples_j_plus,next_samples_k_plus,next_samples_i_minus,next_samples_j_minus, next_samples_k_minus))

    return verts, faces, samples, next_indices
