from mindspore.train.serialization import load_checkpoint, save_checkpoint


def restore_ckpt(load_ckpt_path, save_ckpt_path)
param_dict = load_checkpoint(args_opt.load_ckpt_path)