from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.cityscapes import CityscapesDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

if __name__ == "__main__":
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 8
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = False
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21_modified.yaml').cpu()


    state_dict = load_state_dict(resume_path, location='cpu')

    # 删除不匹配的权重
    if 'control_model.input_hint_block.0.weight' in state_dict:
        del state_dict['control_model.input_hint_block.0.weight']

    # 加载修改后的权重
    model.load_state_dict(state_dict, strict=False)

    # model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)


    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    dataset = CityscapesDataset()
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(default_root_dir='work_dir/cityscapes_hr',gpus=[6], max_steps=100000, 
                        callbacks=[logger, 
                                    ModelCheckpoint(dirpath='work_dir/cityscapes_hr/ckpt_hr',
                                    save_last=True, every_n_train_steps=5000, save_top_k=-1)],
                        enable_progress_bar=True
                        )


    # Train!
    trainer.fit(model, dataloader)