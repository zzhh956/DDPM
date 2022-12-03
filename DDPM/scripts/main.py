from pathlib import Path
from parse_argue import parser
from pytorch_lightning import Trainer
from diffusion import DiffusionTrainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    arg = parser.parse_args()

    if arg.train:
        print('----------train model----------')

        # ckpt_path = Path('../train_ckpt/resnet18_epoch=' + arg.epoch_idx + '.ckpt')
        # ckpt_path = './train_params/resnet50_epoch=06.ckpt'
        # model = Image_self_supervise.load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        model = DiffusionTrainer()

        ckpt_dir_path = Path('../train_ckpt/')
        params_callback = ModelCheckpoint(dirpath = ckpt_dir_path, filename = 'diffusion'+'_{epoch:02d}', save_top_k = 1, mode = "min", monitor = "avg_loss")
            
        trainer = Trainer(callbacks = [params_callback], accelerator = "gpu", max_epochs = 100)
        # trainer = Trainer(accelerator = "gpu", min_epochs = 200, max_epochs = 250)
        trainer.fit(model)

    elif arg.sample:
        print('----------sample model----------')

        ckpt_path = Path('../train_ckpt/diffusion_epoch=' + arg.epoch_idx + '.ckpt')
        model = DiffusionTrainer(grid=False).load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        
        trainer = Trainer(accelerator = "gpu")
        trainer.test(model)

    elif arg.grid:
        print('----------grid_sample model----------')

        ckpt_path = Path('../train_ckpt/diffusion_epoch=' + arg.epoch_idx + '.ckpt')
        model = DiffusionTrainer(grid=True).load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        
        trainer = Trainer(accelerator = "gpu")
        trainer.test(model)
