from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  
  # # Set the device to GPU if available
  # config.device = torch.device('cuda:1')  # or config.device = 'cuda:1'
  
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  
  training.batch_size = 64
  training.sample_size = 4
  training.epochs = 300 # 13944/3486
  training.snapshot_freq = 10
  training.log_freq = 10
  training.eval_freq = 50

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # evaluation
  config.eval.batch_size = 64
  
  # data
  data = config.data
  data.dataset = 'IXISliced'
  # data.root = '/home/yuchenliu/Dataset/IXI/t1_np/t1_np_sliced_128' # Not correctly normalized
  data.root = '/home/yuchenliu/Dataset/IXI/t1_np_masked_128'
  data.is_complex = False
  data.is_multi = False
  data.image_size = 128
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.num_scales = 2000 # Diffusion steps
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config