import argparse
import math
import signal
from os.path import join
from tqdm import tqdm
from utils import *
import torch.utils.data as data_utils
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = join(os.path.dirname(os.path.realpath(__file__)), '..')
parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
parser.add_argument("--Description", default=[ "This experiment learns a shape representation for planes ",
                    "using data from ShapeNet version 2." ])
parser.add_argument("--log_dir", default=root + '/logs/DeepSDF_' + time.strftime('%m_%d_%H%M%S'))
parser.add_argument("--LogFrequency", default=10)
# data para
parser.add_argument("--DataSource", default="data")
parser.add_argument("--TrainSplit", default="examples/splits/sv2_planes_train.json")
# parser.add_argument("--TrainSplit", default="data/train.json")
parser.add_argument("--TestSplit", default="data/test.json")
parser.add_argument("--SamplesPerScene", default=16384)
parser.add_argument("--ScenesPerBatch", default=8)
parser.add_argument("--DataLoaderThreads", default=16)
# networks para
parser.add_argument("--NetworkArch", default="deep_sdf_decoder")
parser.add_argument("--NetworkSpecs", default={
    "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512],
    "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.2,
    "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in" : [4],
    "xyz_in_all" : False,
    "use_tanh" : False,
    "latent_dropout" : False,
    "weight_norm" : True
    })
parser.add_argument("--CodeLength", default=256)
# optimize para
parser.add_argument("--NumEpochs", default=4000)
parser.add_argument(
    "--batch_split",
    dest="batch_split",
    default=1,
    help="This splits the batch into separate subbatches which are "
    + "processed separately, with gradients accumulated across all "
    + "subbatches. This allows for training with large effective batch "
    + "sizes in memory constrained environments.",
)
parser.add_argument("--SnapshotFrequency", default=1000)
parser.add_argument("--AdditionalSnapshots", default=[ 100, 500 ])
parser.add_argument("--LearningRateSchedule", default=[
    {
      "Type" : "Step",
      "Initial" : 0.0005,
      "Interval" : 500,
      "Factor" : 0.5
    },
    {
      "Type" : "Step",
      "Initial" : 0.001,
      "Interval" : 500,
      "Factor" : 0.5
    }])
parser.add_argument("--ClampingDistance", default=0.1)
parser.add_argument("--CodeRegularization", default=True)
parser.add_argument("--CodeRegularizationLambda", default=1e-4)
parser.add_argument("--CodeBound", default=1)
parser.add_argument("--grad_clip", default=None)
parser.add_argument("--CodeInitStdDev", default=1)
parser.add_argument("--enforce_minmax", default=True)




def main_function():


    signal.signal(signal.SIGINT, signal_handler)


    specs = parser.parse_args()


    # Create logging files/folders for losses
    experiment_directory = specs.log_dir
    os.makedirs(experiment_directory, exist_ok=True)
    log_frequency = specs.LogFrequency
    epoch_log = open(join(experiment_directory, 'train_test.csv'), 'w')
    print('epoch,loss,accuracy', file=epoch_log, flush=True)
    training_log = open(join(experiment_directory, 'train_iterations.csv'), 'w')
    print('epoch,iteration,loss', file=training_log, flush=True)
    with open(join(experiment_directory, 'args.json'), 'w') as f:
        json.dump(dict(vars(specs)), f)


    # load data
    data_source = specs.DataSource
    train_split_file = specs.TrainSplit
    num_samp_per_scene = specs.SamplesPerScene
    scene_per_batch = specs.ScenesPerBatch
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    sdf_dataset = utils.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )
    num_data_loader_threads = specs.DataLoaderThreads
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    sdf_loader_reconstruction = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    num_scenes = len(sdf_dataset)
    print('Datasets loaded. ({} in sdf dataset)'.format(num_scenes))



    # Create decoder
    latent_size = specs.CodeLength
    arch = __import__("networks." + specs.NetworkArch, fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs.NetworkSpecs).to(device)
    print(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )


    # Initialize obj
    code_bound = specs.CodeBound
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        specs.CodeInitStdDev / math.sqrt(latent_size),
    )
    print(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )



    # LOSS AND OPTIMIZER
    lr_schedules = get_learning_rate_schedules(specs)
    loss_l1 = torch.nn.L1Loss(reduction="sum")
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )
    grad_clip = specs.grad_clip
    enforce_minmax = specs.enforce_minmax
    clamp_dist = specs.ClampingDistance
    minT = -clamp_dist
    maxT = clamp_dist
    do_code_regularization = specs.CodeRegularization
    code_reg_lambda = specs.CodeRegularizationLambda


    # Run training iterations to optimize weights
    iteration = 0
    training_loop = tqdm(range(specs.NumEpochs))
    for epoch in training_loop:


        # TRAINING LOOP
        decoder.train()
        loss_epoch = 0
        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        for sdf_data, indices in sdf_loader:
            sdf_data = sdf_data.reshape(-1, 4).to(device)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
            xyz = torch.chunk(xyz, specs.batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                specs.batch_split,
            )
            sdf_gt = torch.chunk(sdf_gt, specs.batch_split)
            batch_loss = 0.0
            optimizer_all.zero_grad()
            for i in range(specs.batch_split):
                batch_vecs = lat_vecs(indices[i]).to(device)
                input = torch.cat([batch_vecs, xyz[i]], dim=1)
                pred_sdf = decoder(input)
                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + reg_loss.cuda()
                chunk_loss.backward()
                batch_loss += chunk_loss.item()
                loss_epoch += batch_loss
            print('%d,%d,%.5f' % (epoch, iteration, batch_loss), file=training_log, flush=True)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer_all.step()
            iteration += 1
        loss_epoch /= len(sdf_dataset)
        print('%d,%.10f,%.10f' % (epoch, loss_epoch, loss_epoch), file=epoch_log, flush=True)



        if epoch % log_frequency == 0:
            utils.plot_all_iterations(experiment_directory)
            utils.plot_train_test([experiment_directory])
            torch.save(
                {'weights': decoder.state_dict(), 'epoch': epoch, 'train_loss': loss_epoch, 'latent_codes':lat_vecs.state_dict()},
                join(experiment_directory, 'experiment.pth'))
            # store reconstructions
            for ith in range(0,1000,100):
                sdf_data, indices = sdf_dataset.__getitem__(ith)
                dirs = experiment_directory + '/reconstruction/' + str(ith) + '/'
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                latent = lat_vecs(indices).to(device)
                mesh_filename = dirs + str(epoch)
                with torch.no_grad():
                    create_mesh(decoder, latent.squeeze(), 32, output_mesh=False, filename=mesh_filename)
            # for ith, data in enumerate(sdf_loader_reconstruction):
            #     if ith < 3:
            #         sdf_data, indices = data
            #         dirs = experiment_directory + '/reconstruction/' + str(ith) + '/'
            #         if not os.path.exists(dirs):
            #             os.makedirs(dirs)
            #         obj = lat_vecs(indices).to(device)
            #         mesh_filename = dirs + str(epoch)
            #         with torch.no_grad():
            #             create_mesh(decoder, obj.squeeze(), 32, output_mesh=False, filename=mesh_filename)



        training_loop.set_description('%.4f' % (float(loss_epoch)))


    training_log.close()
    epoch_log.close()
    print('Done.')





if __name__ == "__main__":


    main_function()
