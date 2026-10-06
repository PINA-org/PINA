"""Utility class for aggregating multiple dataloaders into a single iterable."""


class MultiLoader:
    """
    Aggregate multiple dataloaders into a unified iterable object.

    The multi-loader combines batches produced by multiple dataloaders
    according to the selected batching strategy. It is primarily used to
    coordinate the iteration of multiple training conditions within a single
    training loop.
    """

    def __init__(self, dataloaders):
        """
        Initialization of the :class:`MultiLoader` class.

        :param dict[str, DataLoader] dataloaders: The mapping between
            condition names and their corresponding dataloaders.
        """
        # Initialize attributes
        self.dataloaders = dataloaders

    def __len__(self):
        """
        Return the length of the aggregated dataloader.

        The length is determined by the number of iterations required to
        exhaust the dataloaders, i.e. the maximum length among the aggregated
        dataloaders.

        :return: The length of the aggregated dataloader.
        :rtype: int
        """
        return max(len(dl) for dl in self.dataloaders.values())

    def __iter__(self):
        """
        Iterate over the aggregated dataloaders.

        At each iteration, a dictionary containing one batch per dataloader is
        yielded. If a dataloader is exhausted before the others, its iterator is
        restarted automatically to ensure continuous batch generation.

        :yield: The dictionary mapping each condition name to its batch.
        :rtype: Iterator[dict[str, Any]]
        """
        # Initialize iterators for each dataloader
        iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}

        # Iterate until the maximum number of iterations is reached
        for _ in range(len(self)):
            batch = {}

            # Generate a batch for each dataloader
            for name, dataloader in self.dataloaders.items():

                # Attempt to get the next batch from the dataloader's iterator
                try:
                    batch[name] = next(iterators[name])

                # Restart the iterator if it is exhausted
                except StopIteration:
                    iterators[name] = iter(dataloader)
                    batch[name] = next(iterators[name])

            yield batch
