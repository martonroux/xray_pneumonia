from typing import List
import torch
from random import Random


ListTensors = List[torch.Tensor]


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
        self._check_input_data(normal, "normal")
        self._check_input_data(pneumonia, "pneumonia")

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
    # Initial data checks
    ##
    def _check_input_data(self, data: ListTensors, arg_name: str) -> None:
        if type(data) != list:
            raise ValueError(arg_name + " argument needs to be of type List[torch.Tensor].")
        for val in data:
            if type(val) != torch.Tensor:
                raise ValueError(arg_name + " elements need to be all torch.Tensor")

    ##
    # Iterator functions
    ##
    def __len__(self):
        return len(self._batch)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._batch) == self._batch_idx:  # If reached last batch, raise stop
            self._batch_idx = 0
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
        print("1")
        shuffled_list = self._shuffle_list(seed, shuffled_list)
        print("2")
        labels: list = self._get_labels()
        print("3")
        labels = self._shuffle_list(seed, labels)
        print("4")

        self._batch = []
        for i in range(len(shuffled_list) // self._batch_size):
            print(i, "/", len(shuffled_list) // self._batch_size)
            start = self._batch_size * i
            end = self._batch_size * (i + 1)
            # Append a tuple containing:
            #   0: the list of images of the given batch
            #   1: the labels of said images
            self._batch.append(
                (torch.stack(shuffled_list[start:end], dim=0),
                torch.Tensor(labels[start:end]))
            )
        print("5")
        if len(shuffled_list) % self._batch_size != 0:
            # Append a tuple containing:
            #   0: the list of images of the given batch
            #   1: the labels of said images
            start = -1 * (len(shuffled_list) % self._batch_size)
            self._batch.append(
                (torch.stack(shuffled_list[start:], dim=0),
                torch.Tensor(labels[start:]))
            )
        print("6")

    ##
    # This function allows the user to change the batch size without needing to create a new Dataset
    ##
    def change_batch_size(self, new_batch_size: int, shuffle_seed: int) -> None:
        self._batch_size = new_batch_size
        self.shuffle(shuffle_seed)


if __name__ == "__main__":
    # Small test. File shouldn't run in main anyway
    dataset = Dataset([
        torch.Tensor([123]),
        torch.Tensor([134]),
        torch.Tensor([145]),
        torch.Tensor([156])
    ], [
        torch.Tensor([223]),
        torch.Tensor([234]),
        torch.Tensor([245])
    ], 0, 1, 2)

    for (X, y) in dataset:
        print("X:", X, "\ty:", y)

    dataset.change_batch_size(3, 0)
    print("\nChanged batch size")
    for (X, y) in dataset:
        print("X:", X, "\ty:", y)

