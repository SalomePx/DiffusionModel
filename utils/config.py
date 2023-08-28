import json
import os

from pathlib import Path


class Config:
    def __init__(self, args, global_parameters, **kwargs):

        self.dataset = global_parameters['dataset']
        self.basedir = global_parameters['basedir']
        self.name_exp = global_parameters['name_exp']
        self.savedir = os.path.join(self.basedir, self.name_exp)

        self.dropout = args.dropout
        self.epochs = args.epochs
        self.train_loss = 'mae'
        self.iters = args.iters
        self.lr = args.lr
        self.batch_size = args.batch

        # self.train_checkpoint = 'weights_best.h5'
        # self.train_checkpoint_last = 'weights_last.h5'
        # self.train_checkpoint_epoch = 'weights_now.h5'

        self.update_parameters(**kwargs)
        self.save_json()

    def update_parameters(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def save_json(self):
        config_file = os.path.join(self.savedir, 'config.json')
        savedir = Path(self.savedir)
        savedir.mkdir(parents=True, exist_ok=True)

        vars_config, fpath = vars(self), str(config_file)

        with open(fpath, 'w') as f:
            f.write(json.dumps(vars_config))
