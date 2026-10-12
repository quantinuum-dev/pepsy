"""Evaluation-local counters with no tensor reads or persistent state."""


class MaterializationStats:
    def __init__(self):
        self.products_requested = 0
        self.products_evaluated = 0
        self.exponentials_requested = 0
        self.batches = []
        self.lower_contractions = 0
        self.lower_requested = 0
        self.partition_products = 0

    def report(self, active):
        evaluated = sum(self.batches)
        return dict(
            local_products_requested=self.products_requested,
            local_products_evaluated=self.products_evaluated,
            local_products_reused=self.products_requested - self.products_evaluated,
            factor_exponentials_requested=self.exponentials_requested,
            factor_exponentials_evaluated=evaluated,
            factor_exponentials_reused=self.exponentials_requested - evaluated,
            exponential_batch_sizes=tuple(self.batches),
            lower_contractions=self.lower_contractions,
            lower_contractions_requested=self.lower_requested,
            lower_contractions_reused=self.lower_requested - self.lower_contractions,
            partition_products=self.partition_products,
            active_blocks=active.active_block_count,
            dense_nbytes=active.dense_nbytes,
        )


class EvaluationProductData(tuple):
    def __new__(cls, factors, stats=None, *, is_factor=False):
        obj = super().__new__(cls, factors)
        obj.stats = stats
        obj.is_factor = is_factor
        return obj
