from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = np.clip(input_sdf, -self.truncation_distance, self.truncation_distance)
        target_df = np.clip(target_df, 0, self.truncation_distance)
        
        
        # TODO Stack (distances, sdf sign) for the input sdf
        input_sdf = np.stack([np.abs(input_sdf), np.sign(input_sdf)])
        
        # TODO Log-scale target df
        target_df = np.log(target_df + 1)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        shape_sdf_path = Path(ShapeNet.dataset_sdf_path / f"{shapenet_id}.sdf")
        with open(shape_sdf_path,"rb") as f:
            byte_string = f.read()
            # uint64 -> 8 bytes, signed=False by default
            dim_x = int.from_bytes(byte_string[:8], "little")
            dim_y = int.from_bytes(byte_string[8:16], "little")
            dim_z = int.from_bytes(byte_string[16:24], "little")
            sdf = np.frombuffer(byte_string[24:], dtype=np.float32)
        return sdf.reshape(dim_x, dim_y, dim_z)

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        shape_df_path = Path(ShapeNet.dataset_df_path / f"{shapenet_id}.df")
        with open(shape_df_path,"rb") as f:
            byte_string = f.read()
            # uint64 -> 8 bytes, signed=False by default
            dim_x = int.from_bytes(byte_string[:8], "little")
            dim_y = int.from_bytes(byte_string[8:16], "little")
            dim_z = int.from_bytes(byte_string[16:24], "little")
            df = np.frombuffer(byte_string[24:], dtype=np.float32)
        return df.reshape(dim_x, dim_y, dim_z)
