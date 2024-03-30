import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('..')
sys.path.append('../utils')
import argparse
import signal
from tqdm import tqdm
from train.train_deep_sdf import parser as sdf_parser
from utils import *
import bempp
from bempp.api.linalg import gmres
from bempp.api.operators.far_field import helmholtz as helmholtz_far



root = join(os.path.dirname(os.path.realpath(__file__)), '..')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="")
parser.add_argument("--sdf_log_dir", default=root + '/logs/DeepSDF_plane')
parser.add_argument("--optimization_dir", default=root + '/logs/opt_plane_' + time.strftime('%m_%d_%H%M%S'))
parser.add_argument("--LogFrequency", default=1)
parser.add_argument("--iterations", default=2000)
parser.add_argument("--wavenumber", default=5*np.pi)
parser.add_argument("--num_d", default=10)
parser.add_argument("--num_far", default=100)
parser.add_argument("--z_i", type=int, default=410) # 0-1780 in train
parser.add_argument("--z_t", type=int, default=2230) # 1780-2236 in test
parser.add_argument("--delta", default=0)
parser.add_argument("--CodeRegularization", default=False)
parser.add_argument("--CodeRegularizationLambda", default=5e-1)
parser.add_argument("--tol", default=1E-4)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--Nr", type=int, default=32)
parser.add_argument("--level", default=0.05)




def main_function():


    signal.signal(signal.SIGINT, signal_handler)


    # Parse input arguments
    args = parser.parse_args()
    sdf_args = sdf_parser.parse_args()


    # Create logging files/folders for losses
    optimization_meshes_dir = args.optimization_dir
    os.makedirs(optimization_meshes_dir, exist_ok=True)
    images_dir = optimization_meshes_dir + '/intermediate'
    os.makedirs(images_dir, exist_ok=True)
    label_dir = optimization_meshes_dir + '/label'
    os.makedirs(label_dir, exist_ok=True)
    epoch_log = open(join(optimization_meshes_dir, 'optimization.csv'), 'w')
    print('epoch,loss,indicator error', file=epoch_log, flush=True)
    with open(join(optimization_meshes_dir, 'args.json'), 'w', newline='\n') as f:
        json.dump(dict(vars(args)), f, indent=1)


    # load decoder and latent
    arch = __import__("networks." + sdf_args.NetworkArch, fromlist=["Decoder"])
    decoder = arch.Decoder(sdf_args.CodeLength, **sdf_args.NetworkSpecs).to(device)  # for plane
    saved_model_state = torch.load(args.sdf_log_dir + '/experiment_train.pth', map_location=torch.device('cpu'))
    decoder.load_state_dict(saved_model_state['weights'])
    latent = saved_model_state["latent_codes"]["weight"].to(device)


    # initialize and visualize initialization
    latent_init = latent[args.z_i]
    latent_init.requires_grad = True
    verts_init, faces_init, normals_init = utils.create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr, l=args.level)
    image_filename = os.path.join(images_dir, "init.html")
    write_verts_faces_to_file(verts_init, faces_init, image_filename)
    torch.save(latent_init, label_dir + '/' + "init.pt")
    print('init vert shape:', verts_init.shape, '\ninit faces shape:', faces_init.shape)


    # load decoder and latent
    arch = __import__("networks." + sdf_args.NetworkArch, fromlist=["Decoder"])
    decoder_total = arch.Decoder(sdf_args.CodeLength, **sdf_args.NetworkSpecs).to(device)  # for plane
    saved_model_state_total = torch.load(args.sdf_log_dir + '/experiment_total.pth', map_location=torch.device('cpu'))
    decoder_total.load_state_dict(saved_model_state_total['weights'])
    latent_total = saved_model_state_total["latent_codes"]["weight"].to(device)


    # target stuff
    latent_target = latent_total[args.z_t]
    latent_target.requires_grad = False
    verts_target, faces_target, normals_target = utils.create_mesh_with_edge(decoder_total, latent_target, N=args.Nr, l=args.level)
    image_filename = os.path.join(images_dir, "target.html")
    write_verts_faces_to_file(verts_target, faces_target, image_filename)
    torch.save(latent_target, label_dir + '/' + "target.pt")
    indicator_target = indicator_plane(decoder_total, latent_target, N=args.Nr, l=args.level)
    print('latent.shape: ', latent.shape, '\nlatent_total.shape: ', latent_total.shape)


    # wavenumber
    k = args.wavenumber
    # direction of incident field
    num_d = args.num_d
    phi = (np.sqrt(5) - 1) / 2
    n = np.array([np.arange(0, num_d)])
    z = ((2 * n + 1) / num_d - 1)
    x = (np.sqrt(1 - z ** 2)) * np.cos(2 * np.pi * (n + 1) * phi)
    y = (np.sqrt(1 - z ** 2)) * np.sin(2 * np.pi * (n + 1) * phi)
    d_i = np.concatenate((x, y, z), axis=0).transpose()
    # direction of Far field data
    num_far = args.num_far
    phi = (np.sqrt(5) - 1) / 2
    n = np.array([np.arange(0, num_far)])
    z = ((2 * n + 1) / num_far - 1)
    x = (np.sqrt(1 - z ** 2)) * np.cos(2 * np.pi * (n + 1) * phi)
    y = (np.sqrt(1 - z ** 2)) * np.sin(2 * np.pi * (n + 1) * phi)
    points = np.concatenate((x, y, z), axis=0)



    # far filed pattern of target object
    far_target = np.zeros((d_i.shape[0], points.shape[1])).astype(complex)
    grid = bempp.api.Grid(verts_target.transpose(), faces_target.transpose())
    piecewise_const_space = bempp.api.function_space(grid, "DP", 0)
    identity = bempp.api.operators.boundary.sparse.identity(
        piecewise_const_space, piecewise_const_space, piecewise_const_space)
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space, k)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space, k)
    lhs = 0.5 * identity + adlp - 1j * k * slp
    single_far = helmholtz_far.single_layer(piecewise_const_space, points, k)
    for ith in range(d_i.shape[0]):
        @bempp.api.complex_callable
        def combined_ui(x, n, domain_index, result):
            result[0] = 1j * k * np.exp(1j * k * (x[0] * d_i[ith][0] + x[1] * d_i[ith][1] + x[2] * d_i[ith][2])) * (
                        n[0] * d_i[ith][0] + n[1] * d_i[ith][1] + n[2] * d_i[ith][2] - 1)
        combined_ui_fun = bempp.api.GridFunction(piecewise_const_space, fun=combined_ui)
        un_fun, info = gmres(lhs, combined_ui_fun, tol=args.tol)
        far_target[ith,:] = -single_far * un_fun
    normal = np.random.rand(d_i.shape[0], points.shape[1])
    # far_target = far_target + args.delta*normal*far_target
    far_target = far_target + args.delta * normal * np.max(far_target)



    # optimizer
    # optimizer = torch.optim.Adam([latent_init], lr=args.lr)
    optimizer = torch.optim.SGD([latent_init], lr=args.lr)



    do_code_regularization = args.CodeRegularization
    code_reg_lambda = args.CodeRegularizationLambda


    print("Starting optimization:")
    training_loop = tqdm(range(args.iterations))
    for epoch in training_loop:

        optimizer.zero_grad()

        # first extract iso-surface
        verts, faces, normals = utils.create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr, l=args.level)


        grid = bempp.api.Grid(verts.transpose(), faces.transpose())
        piecewise_const_space = bempp.api.function_space(grid, "DP", 0)


        # far field pattern and un_value
        far = np.zeros((d_i.shape[0], points.shape[1])).astype(complex)
        un_value = np.zeros((d_i.shape[0], verts.shape[0])).astype(complex)
        identity = bempp.api.operators.boundary.sparse.identity(
            piecewise_const_space, piecewise_const_space, piecewise_const_space)
        adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
            piecewise_const_space, piecewise_const_space, piecewise_const_space, k)
        slp = bempp.api.operators.boundary.helmholtz.single_layer(
            piecewise_const_space, piecewise_const_space, piecewise_const_space, k)
        lhs = 0.5 * identity + adlp - 1j * k * slp
        single_far = helmholtz_far.single_layer(piecewise_const_space, points, k)
        for ith in range(d_i.shape[0]):
            @bempp.api.complex_callable
            def combined_ui(x, n, domain_index, result):
                result[0] = 1j * k * np.exp(1j * k * (x[0] * d_i[ith][0] + x[1] * d_i[ith][1] + x[2] * d_i[ith][2])) * (
                        n[0] * d_i[ith][0] + n[1] * d_i[ith][1] + n[2] * d_i[ith][2] - 1)
            combined_ui_fun = bempp.api.GridFunction(piecewise_const_space, fun=combined_ui)
            un_fun, info = gmres(lhs, combined_ui_fun, tol=args.tol)
            far[ith, :] = -single_far * un_fun
            un_value[ith, :] = un_fun.evaluate_on_vertices().squeeze()


        # adjoint equation for wn_value
        wn_value = np.zeros((d_i.shape[0], verts.shape[0])).astype(complex)
        for ith in range(d_i.shape[0]):
            # adjoint equation
            @bempp.api.complex_callable
            def combined_wi(x, normal, domain_index, result):
                out = 0
                for jth in range(points.shape[1]):
                    out = out - 1j * k * np.conj(far[ith, jth] - far_target[ith, jth]) * np.exp(
                        - 1j * k * (x[0] * points[0, jth] + x[1] * points[1, jth] + x[2] * points[2, jth])) * (
                                  x[0] * normal[0] + x[1] * normal[1] + x[2] * normal[2] + 1)
                result[0] = out / (points.shape[1] * 4 * np.pi)
            combined_wi_fun = bempp.api.GridFunction(piecewise_const_space, fun=combined_wi)
            wn_fun, info = gmres(lhs, combined_wi_fun, tol=args.tol)
            wn_value[ith, :] = wn_fun.evaluate_on_vertices().squeeze()


        # shape_derivative
        shape_derivative = torch.zeros(verts.shape[0])
        for ith in range(d_i.shape[0]):
            shape_derivative = shape_derivative + torch.from_numpy(un_value[ith, :] * wn_value[ith, :]).to(device)
        shape_derivative = shape_derivative / d_i.shape[0]


        # loss Back-propagating to mesh vertices
        normals_upstream = torch.tensor(normals.astype(float), requires_grad=False, dtype=torch.float64,
                                               device=device)
        dL_dx_i = torch.real(shape_derivative.unsqueeze(1)) * normals_upstream


        # compute loss of far field pattern
        loss_epoch = np.linalg.norm(far - far_target) ** 2 / (2 * points.shape[1] * d_i.shape[0])
        indicator_init = indicator_plane(decoder, latent_init, N=args.Nr, l=args.level)
        indicator_error = torch.norm(indicator_target-indicator_init).squeeze().detach().cpu().numpy()
        print('%d,%.5f,%.5f' % (epoch, loss_epoch, indicator_error),file=epoch_log, flush=True)
        training_loop.set_description('%.4f' % (float(loss_epoch)))


        """
            mesh vertices Back-propagating to label
        """
        # first compute normals
        optimizer.zero_grad()
        verts_dr = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float64, device=device)
        latent_inputs = latent_init.expand(verts_dr.shape[0], -1)
        pred_sdf = decoder(torch.cat([latent_inputs, verts_dr], 1).to(torch.float64))
        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        # normalization to take into account for the fact sdf is not perfect...
        normals = verts_dr.grad/torch.norm(verts_dr.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()
        dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        if do_code_regularization:
            loss_backward = torch.sum(dL_ds_i * pred_sdf) + code_reg_lambda * torch.norm(latent_init)
        else:
            loss_backward = torch.sum(dL_ds_i * pred_sdf)
        loss_backward.backward()
        # and update params
        optimizer.step()


        # log stuff
        if epoch % args.LogFrequency == 0:
            utils.plot_error(optimization_meshes_dir)
            image_filename = images_dir + '/' + str(epoch) + '.html'
            field = torch.real(shape_derivative.unsqueeze(1))
            write_verts_faces_fields_to_file(verts, faces, field, image_filename)
            torch.save(latent_init.detach(), label_dir + '/' + str(epoch) + ".pt")


    epoch_log.close()


    print("Done.")



if __name__ == "__main__":

    main_function()


