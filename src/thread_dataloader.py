from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import random
import math
import torch

## This was needed due to some weird bug with windows & multiprocessing
class ThreadDataLoader():
  def __init__(self, dataset, batch_size = 1, shuffle = False, num_workers = 2, look_ahead = 2):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.look_ahead = look_ahead

  def __getitem__(self, idx):
    return self.dataset[idx]

  def __len__(self):
    return math.ceil(len(self.dataset) / self.batch_size)

  def __iter__(self):
    executor = PoolExecutor(max_workers=self.num_workers)
    idxs = list(range(len(self.dataset)))
    if self.shuffle:
      random.shuffle(idxs)
    look_ahead = self.look_ahead * max(self.batch_size, self.num_workers)
    futures = {i: executor.submit(ThreadDataLoader.__getitem__, self, i) for i in idxs[:look_ahead]}
    collated_arr = []
    for i, idx in enumerate(idxs):
      data = futures[idx].result()
      del futures[idx]
      if i + look_ahead < len(idxs):
        futures[idxs[i + look_ahead]] = executor.submit(ThreadDataLoader.__getitem__, self, idxs[i + look_ahead])
      collated_arr.append(data)
      collated_batch = []
      if len(collated_arr) == self.batch_size:
          collated_batch = [torch.stack(data) for data in zip(*collated_arr)]
          yield collated_batch
          collated_arr = []
    if len(collated_arr) > 0:
      collated_batch = [torch.stack(data) for data in zip(*collated_arr)]
      yield collated_batch
