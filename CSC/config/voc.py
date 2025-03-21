
def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=64, type=int)
    subparsers.add_argument('--epochs', default=20, type=int)

    # Continual learning parameters
    subparsers.add_argument('--base_classes', default=5, type=int, help='classes of base task')
    subparsers.add_argument('--task_size', default=3, type=int, help='classes of sequential tasks')
    subparsers.add_argument('--total_classes', default=20, type=int, help='total classes(voc:20)')

    # Replay parameters
    subparsers.add_argument('--replay', default=False, type=bool,)
    subparsers.add_argument('--buffer_size', default=40, type=int, help='buffer_size = num_protos * total_classes(voc40)')
    subparsers.add_argument('--num_protos', default=2, type=int, help= 'numbers of each class')
    subparsers.add_argument('--replay_alpha', default=0.5, type=int, help= 'parameter of replay(default=0.5)')
    subparsers.add_argument('--sample_method', default='herding', type=str,help='sample method, herding')

    # Model parameters
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--alpha', default=0.5, type=int, help= 'parameter of kd(default=0.15)')
    subparsers.add_argument('--beta', default=-0.04, type=int, help= 'parameter of Max-Entropy(default=-0.04)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--lr', type=float, default=0.00004, metavar='LR', help='learning rate')

    # Data parameters
    subparsers.add_argument('--root_dir', default="./src/datasets/VOC2007/VOCdevkit/VOC2007", type=str, help='root_dir')
    subparsers.add_argument('--dataset_name', default='voc', type=str, help='dataset name')
    subparsers.add_argument('--output_dir', default='./checkpoints', help='path where to save, empty for no saving')
    subparsers.add_argument('--seed', default=40, type=int)
    subparsers.add_argument('--num_workers', default=32, type=int)


