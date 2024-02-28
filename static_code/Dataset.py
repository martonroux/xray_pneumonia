from typing import List
import torch
from random import Random


ListTensors = List[torch.tensor]


class Dataset:
    def __init__(self,
                 normal: ListTensors,  # List containing Tensors for each Normal image
                 pneumonia: ListTensors,  # Equivalent of `normal` for Pneumonia images
                 normal_label: int,  # Label to use for Normal data (0 or 1)
                 pneumonia_label: int,  # Label to use for Pneumonia data (0 or 1, opposite of `normal_label`)
                 batch_size: int):
        self._norm = normal
        self._pneu = pneumonia
        self._norm_label = normal_label
        self._pneu_label = pneumonia_label
        self._batch_size = batch_size

        if self._norm_label == self._pneu_label:
            raise ValueError("Given labels for Normal and Pneumonia data are identical.")

        # _batch will contain the current batch, according to the current shuffle, for the iterator.
        # It is organised as such:
        # List of tuples. Each tuple represents a batch.
        # In each tuple, you have, at index:
        #   0: list of images of the given batch
        #   1: list of labels of said images
        self._batch: list = []
        self._batch_idx: int = 0

        # Run a shuffle with seed 0 to initialize batch
        self.shuffle(0)

    ##
    # Iterator functions
    ##
    def __iter__(self):
        return self

    def __next__(self):
        if len(self._batch) == self._batch_idx:  # If reached last batch, raise stop
            raise StopIteration
        self._batch_idx += 1
        return self._batch[self._batch_idx - 1]

    def __getitem__(self, item):
        return self._batch[item]

    ##
    # Batching functions
    ##
    def _get_shuffle_list(self) -> list:
        ##
        # Initializes the list that will be shuffled (contains both normal and pneumonia data)
        ##
        shuffle_list: list = list(self._norm)
        for pneu in self._pneu:
            shuffle_list.append(pneu)
        return shuffle_list

    def _get_labels(self) -> list:
        ##
        # Initializes the list of labels
        ##
        labels = [self._norm_label for _ in range(len(self._norm))]
        for _ in range(len(self._pneu)):
            labels.append(self._pneu_label)
        return labels

    def _shuffle_list(self, seed: int, to_shuffle: list) -> list:
        ##
        # Shuffles a given list `to_shuffle` according to the seed `seed` using the `random` python library
        ##
        rand = Random(seed)
        rand.shuffle(to_shuffle)
        return to_shuffle

    def shuffle(self, seed: int):
        ##
        # Main function to shuffle and generate a batch. Shuffles all data according to a given seed and modifies
        # the batch with the newly shuffled data.
        ##
        shuffled_list: list = self._get_shuffle_list()
        shuffled_list = self._shuffle_list(seed, shuffled_list)
        labels: list = self._get_labels()
        labels = self._shuffle_list(seed, labels)

        batches = []
        for i in range(len(shuffled_list) // self._batch_size):
            start = self._batch_size * i
            end = self._batch_size * (i + 1)
            # Append a tuple containing:
            #   0: the list of images of the given batch
            #   1: the labels of said images
            batches.append(
                (torch.stack(shuffled_list[start:end], dim=0),
                torch.Tensor(labels[start:end]))
            )

        if len(shuffled_list) % self._batch_size != 0:
            # Append a tuple containing:
            #   0: the list of images of the given batch
            #   1: the labels of said images
            start = -1 * (len(shuffled_list) % self._batch_size)
            batches.append(
                (torch.stack(shuffled_list[start:], dim=0),
                torch.Tensor(labels[start:]))
            )

        self._batch = batches


if __name__ == "__main__":
    # Small test. File shouldn't run in main anyway
    dataset = Dataset(["norm1", "norm2", "norm3", "norm4"], ["pneu1", "pneu2", "pneu3"], 0, 1, 2)
    for (X, y) in dataset:
        print("X:", X, "\ty:", y)
