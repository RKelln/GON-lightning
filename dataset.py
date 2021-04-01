# Based on code from:
# - https://github.com/npuichigo/pytorch_lmdb_dataset
# - https://github.com/forwchen/vid2frame

import numpy as np
from io import BytesIO
from torch.utils.data import Dataset
from os import PathLike

from PIL import Image
from typing import Any, Callable, cast, List, Optional, Tuple, Union

class LmdbDataset(Dataset):
    """Lmdb dataset."""

    def __init__(self, 
        lmdb_path: PathLike, 
        transform: Optional[Callable] = None,
        batch_size: int = 32 
    ) -> None:
        super(LmdbDataset, self).__init__()
        import lmdb
        self.path = lmdb_path
        self.batch_size = batch_size # this is needed to make sure the length is a multiple of batch size
        self.transform = transform
        self.env = lmdb.open(str(lmdb_path), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key, _ in txn.cursor()]

            # get image metadata
            v = txn.get(self.keys[0])
            try:
                img = Image.open(BytesIO(v))
            except Exception as e:
                print(f"LMDBDataset failed to load")
                raise e
            self.img_size = img.size
            self.num_channels = len(img.getbands())
        # HACK: When the batchsize is not an even multiple of the length then training
        #       fails at the end of an epoch
        #self.length = self.length - (self.length % self.batch_size)

    def __getitem__(self, index: int) -> Any:
        with self.env.begin(write=False) as txn:
            v = txn.get(self.keys[index])
            try:
                img = Image.open(BytesIO(v))
            except Exception as e:
                print(f"Reading failed for {self.keys[index]}")
                raise e

        if self.transform is not None:
            return self.transform(img), 0
            
        return img, 0

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return f"{self.path} len: {self.length} size: {self.img_size} chan: {self.num_channels}"