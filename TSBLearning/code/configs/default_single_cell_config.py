
import ml_collections

def get_single_cell_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 4
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  config.t0 = 0
  config.problem_name = 'single_cell'
  config.num_itr = 200
  config.forward_net = 'res' 
  config.backward_net = 'res'
  config.use_arange_t = False 
  config.num_epoch = 1
  config.num_stage = 1
  config.train_bs_x = 100
  config.train_bs_t = 10
  
  config.sde_type = 'tbm' 
  config.dir = 'single_cell/sb'
  # sampling
  config.samp_bs = 100
  config.sigma = 1
  config.diffu = 2

  config.snapshot_freq = 1
  config.use_corrector = True # True
  config.snr = 0.05
  
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-4
  config.lr_gamma = 0.99

  model_configs=None
  return config, model_configs

