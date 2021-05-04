from .base_options import BaseOptions
from pytorch_lightning import Trainer

class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        parser = self.parser
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--nepoch_decay', type=int, default=100, help='# of epochs to linearly decay learning rate to zero')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer to use')
        parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help="learning rate")
        parser.add_argument('--momentum', default=0, type=float)
        parser.add_argument('--weight_decay', default=0, type=float)
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='the number of warmup epochs when using lr policy LinearWarmupCosineAnnealing')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_decay_factor', default=0.1, type=float, help="decay factor to use with multiplicative learning rate schedulers")
        parser.add_argument('--lr_decay_milestones', type=str, default='[500,1000]', help='the milestones at which to decay the learning rate, when using the multi step lr policy')
        parser = Trainer.add_argparse_args(parser)
        self.parser = parser
        self.is_train = True
