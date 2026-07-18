from lightning.pytorch.cli import LightningCLI
from litmodules import MultiTaskLit, MultiTaskDM


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.data_config', 'model.data_config')
        parser.link_arguments('data.train_batch_size', 'model.train_batch_size')


if __name__ == '__main__':
    MyLightningCLI(MultiTaskLit, MultiTaskDM)
