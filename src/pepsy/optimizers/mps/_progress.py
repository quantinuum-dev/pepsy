"""Native replay progress with optional caller-owned display lifetime."""


class _BorrowedProgressBar:
    def __init__(self, display):
        self.display = display

    def __getattr__(self, name):
        return getattr(self.display, name)

    def close(self):
        # A caller can aggregate several slices/probes into one bar. Closing
        # a replay segment must not close that caller's display.
        pass


def make_progress_bar(progress, **options):
    """Reuse an external update/set_postfix display, or create native tqdm."""
    if callable(getattr(progress, "update", None)) and callable(
        getattr(progress, "set_postfix", None)
    ):
        return _BorrowedProgressBar(progress)
    from tqdm import tqdm

    return tqdm(**options)
