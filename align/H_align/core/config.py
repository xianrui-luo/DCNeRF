import argparse



def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers. Default: 12")
net_arg.add_argument(
    "--clusters", type=int, default=500, help=""
    "cluster number in OANet. Default: 500")
net_arg.add_argument(
    "--iter_num", type=int, default=1, help=""
    "iteration number in the iterative network. Default: 1")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer. Default: 128")
net_arg.add_argument(
    "--use_fundamental", type=str2bool, default=False, help=""
    "train fundamental matrix estimation. Default: False")
net_arg.add_argument(
    "--share", type=str2bool, default=False, help=""
    "share the parameter in iterative network. Default: False")
net_arg.add_argument(
    "--use_ratio", type=int, default=0, help=""
    "use ratio test. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--use_mutual", type=int, default=0, help=""
    "use matual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--ratio_test_th", type=float, default=0.8, help=""
    "ratio test threshold. Default: 0.8")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_path", type=str, default='/data1/zhaoweiyue/benchmark-results/', help=""
    "name of the dataset")

# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_num_kp", type=int, default=2000, help=""
    "number of keypoints per image")
obj_arg.add_argument(
    "--obj_top_k", type=int, default=-1, help=""
    "number of keypoints above the threshold to use for "
    "essential matrix estimation. put -1 to use all. ")
obj_arg.add_argument(
    "--obj_geod_type", type=str, default="episym",
    choices=["sampson", "episqr", "episym"], help=""
    "type of geodesic distance")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")


# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--weight_decay", type=float, default=0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--momentum", type=float, default=0.9, help=""
    "momentum")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=1.0, help=""  #0.5
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_A", type=float, default=0.5, help=""
    "weight of the A loss")
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")
loss_arg.add_argument(
    "--geo_loss_margin", type=float, default=0.1, help=""
    "clamping margin in geometry loss")
loss_arg.add_argument(
    "--ess_loss_margin", type=float, default=0.1, help=""
    "clamping margin in contrastive loss")
loss_arg.add_argument(
    "--loss_EpipolarGeometryDis", type=float, default=1.0, help=""
    "clamping margin in EpipolarGeometryDis loss")
# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--load", type=bool, default=False, help=""
    "load model")
train_arg.add_argument(
    "--epochs", type=int, default=250, help=""
    "epochs")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    "--train_batch_size", type=int, default=16, help=""
    "batch size")
train_arg.add_argument(
    "--gpu_id", type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
train_arg.add_argument(
    "--num_processor", type=int, default=8, help='numbers of used cpu')
train_arg.add_argument(
    "--train_iter", type=int, default=1000000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--log_base", type=str, default="./log_NM6/", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--log_suffix", type=str, default="", help=""
    "suffix of log dir")
train_arg.add_argument(
    "--val_intv", type=int, default=20000, help=""
    "validation interval")
train_arg.add_argument(
    "--save_intv", type=int, default=1000, help=""
    "summary interval")
train_arg.add_argument(
    "--method", type=str, default='sift_2048_hardnet', help=""
    "the method of data generation")
train_arg.add_argument(
    "--model_name", type=str, default='model', help=""
    "the method of model")
# -----------------------------------------------------------------------------
# Testing
test_arg = add_argument_group("Test")
test_arg.add_argument(
    "--use_ransac", type=str2bool, default=True, help=""
    "use ransac when testing?")
test_arg.add_argument(
    "--model_path", type=str, default="", help=""
    "saved best model path for test")
test_arg.add_argument(
    "--res_path", type=str, default="", help=""
    "path for saving results")


# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
