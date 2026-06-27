import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Indexes a robomimic HDF5 file and samples fixed-length subsequences.

    Each item returns:
      'image'  : (seq_len, 3, 64, 64) float32 in [0, 1]
      'action' : (seq_len, 7)          float32

    Demos shorter than seq_len are skipped to avoid padding artifacts.

    num_workers=0 is recommended; for >0 workers set the env var
    HDF5_USE_FILE_LOCKING=FALSE and reopen self.f inside a worker_init_fn.
    """

    def __init__(self, hdf5_path: str, seq_len: int = 50):
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.f = h5py.File(hdf5_path, 'r')

        # Build flat index: list of (demo_key, start_idx) for every valid window
        self._index: list[tuple[str, int]] = []
        for key in sorted(self.f['data'].keys()):
            T = int(self.f['data'][key].attrs['num_samples'])
            if T < seq_len:
                continue
            for start in range(T - seq_len + 1):
                self._index.append((key, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key, start = self._index[idx]
        end = start + self.seq_len
        demo = self.f[f'data/{key}']

        # (L, 84, 84, 3) uint8 → (L, 3, 84, 84) float [0,1] → (L, 3, 64, 64)
        imgs = demo['obs/image_wrist'][start:end]  # (L, H, W, 3)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
        imgs = F.interpolate(imgs, size=(64, 64), mode='bilinear', align_corners=False)

        acts = torch.from_numpy(
            demo['actions'][start:end].astype('float32')
        )  # (L, 7)

        return {'image': imgs, 'action': acts}
