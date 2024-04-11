from os.path import join
import numpy as np
import os
import glob
import logging
import random
import torch
import torch.utils.data
from utils import deep_sdf as ws
import torch.utils.data
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge
torch.set_default_tensor_type(torch.DoubleTensor)
import torch_scatter as ts
from torch import LongTensor




class heldata(torch.utils.data.Dataset):

    def __init__(self, root: str, train: bool = True):

        files_all = []
        files_in_class = os.listdir(root)
        files_all += [join(root, x) for x in files_in_class]
        self.files_all = files_all
        names = []
        names += [x.split('.')[0] for x in files_in_class]
        if train:
            self.files = files_all[int(0*len(files_all)):int(0.9*len(files_all))]
            self.names = names[int(0*len(files_all)):int(0.9*len(names))]
        else:
            self.files = files_all[int(0.9 * len(files_all)):int(0.95 * len(files_all))]
            self.names = names[int(0.9 * len(names)):int(0.95 * len(files_all))]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True).item()

        verts = torch.from_numpy(data['verts'])
        faces = torch.from_numpy(data['faces'].astype(np.int16)).long()
        # normals = torch.from_numpy(data['normals'])
        normals = torch.ones((verts.shape[0],3))
        value = torch.cat((torch.from_numpy(np.real(data['value'])).unsqueeze(1), torch.from_numpy(np.imag(data['value'])).unsqueeze(1)), dim=1)
        mesh = FaceToEdge(False)(Data(pos=verts, face=faces.t()))
        edges = mesh.edge_index

        f0 = faces.type(LongTensor)
        f1 = torch.ones(faces.shape[0], 3)
        f1[:, 0] = faces[:, 1]
        f1[:, 1] = faces[:, 0]
        f1[:, 2] = faces[:, 2]
        f1 = f1.type(LongTensor)
        f2 = torch.ones(faces.shape[0], 3)
        f2[:, 0] = faces[:, 2]
        f2[:, 1] = faces[:, 0]
        f2[:, 2] = faces[:, 1]
        f2 = f2.type(LongTensor)
        faces_perm = torch.cat((f0, f1, f2), dim=0)
        tangent1 = verts[faces_perm[:, 1]] - verts[faces_perm[:, 0]]
        tangent2 = verts[faces_perm[:, 2]] - verts[faces_perm[:, 0]]
        area_per_face = tangent1.cross(tangent2).norm(dim=1) / 2
        total_area = ts.scatter_add(area_per_face, faces_perm[:, 0], dim=0,
                                    out=torch.zeros(verts.shape[0]))

        return verts, edges, faces, total_area, normals, value, idx, self.names[idx]







class EITDATA_D(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True):
        files_all = []
        files_in_class = os.listdir(root)
        files_all += [join(root, x) for x in files_in_class]
        self.files_all = files_all
        names = []
        names += [x.split('.')[0] for x in files_in_class]



        num_ver = 0
        for i in range(len(files_all)):
            data = np.load(files_all[i], allow_pickle=True).item()
            verts, faces, observe = data['verts'], data['faces'], data['observe']
            num_ver = max(verts.shape[0], num_ver)
        self.num_vert = num_ver
        self.num_boun = observe.shape[0]


        if train:
            self.files = files_all[int(0*len(files_all)):int(0.01*len(files_all))]
            self.names = names[int(0*len(files_all)):int(0.01*len(names))]
        else:
            self.files = files_all[int(0.8 * len(files_all)):int(0.801 * len(files_all))]
            self.names = names[int(0.8 * len(names)):int(0.801 * len(files_all))]

    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True).item()

        verts = torch.from_numpy(data['verts'])
        faces = torch.from_numpy(data['faces']).long()
        partial = data['observe']

        mesh = FaceToEdge(False)(Data(pos=verts, face=faces.t()))
        edges = mesh.edge_index
        if verts.shape[0]<self.num_vert:
            n = verts.shape[0]
            verts = torch.cat((verts, verts[0, :].repeat(self.num_vert - verts.shape[0], 1)), 0)
            m = verts.shape[0]-2
            for i in range(n, m):
                edges = torch.cat((edges, torch.tensor([[i], [i + 1]])), dim=1)
            edges = torch.cat((edges, torch.tensor([[m+1], [n]])), dim=1)
        return verts.cpu(), edges.cpu(), faces.cpu(), partial.cpu(), idx, self.names[idx]




class EITDATA(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True):
        files_all = []
        files_in_class = os.listdir(root)
        files_all += [join(root, x) for x in files_in_class]
        self.files_all = files_all
        names = []
        names += [x.split('.')[0] for x in files_in_class]
        num_ver = 0
        for i in range(len(files_all)):
            data = np.load(files_all[i], allow_pickle=True).item()
            verts, faces, observe = data['verts'], data['faces'], data['observe']
            num_ver = max(verts.shape[0], num_ver)
        self.num_vert = num_ver
        if train:
            self.files = files_all[int(0*len(files_all)):int(0.8*len(files_all))]
            self.names = names[int(0*len(files_all)):int(0.8*len(names))]
        else:
            self.files = files_all[int(0.8 * len(files_all)):int(0.88 * len(files_all))]
            self.names = names[int(0.8 * len(names)):int(0.88 * len(files_all))]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True).item()

        verts = torch.from_numpy(data['verts'])
        faces = torch.from_numpy(data['faces']).long()
        normals = torch.from_numpy(data['normals'])
        partial = torch.from_numpy(data['observe'][:,3])

        mesh = FaceToEdge(False)(Data(pos=verts, face=faces.t()))
        edges = mesh.edge_index
        # if verts.shape[0]<self.num_vert:
        #     n = verts.shape[0]
        #     verts = torch.cat((verts, verts[0, :].repeat(self.num_vert - verts.shape[0], 1)), 0)
        #     m = verts.shape[0]-2
        #     for i in range(n, m):
        #         edges = torch.cat((edges, torch.tensor([[i], [i + 1]])), dim=1)
        #     edges = torch.cat((edges, torch.tensor([[m+1], [n]])), dim=1)
        return verts.cpu(), edges.cpu(), faces.cpu(), normals.cpu(), partial.cpu(), idx, self.names[idx]








def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename, allow_pickle=True)
    if subsample is None:
        return npz

    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx





