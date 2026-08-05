"""Reusable no-gradient amplitude caching for Torch VMC measurements."""

from __future__ import annotations

from collections import OrderedDict

from .amplitude import _call_amplitude_fn, _unique_config_rows
from .connections import TorchConnections
from .results import _torch_sample_provenance
from ._common import _as_long_matrix
from ..torch_types import _require_torch

__all__ = ["TorchAmplitudeCache"]


class TorchAmplitudeCache:
    """Bounded configuration-to-amplitude cache with model-state invalidation.

    The cache is deliberately detached and no-gradient only. Its provenance
    includes the PEPS object identity, parameter versions, contraction
    settings, dtype, and device, so reusing a cache after an optimization or
    model move cannot silently return stale amplitudes.
    """

    def __init__(self, max_entries=100_000):
        if isinstance(max_entries, bool) or int(max_entries) < 1:
            raise ValueError("max_entries must be a positive integer.")
        self.max_entries = int(max_entries)
        self._values = OrderedDict()
        self._signature = None
        self._requests = 0
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _signature_for(model):
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            parameter_signature = tuple(
                (str(parameter.dtype), str(parameter.device))
                for parameter in parameters()
            )
        else:
            parameter_signature = ()
        return _torch_sample_provenance(model), parameter_signature

    @staticmethod
    def _key(config):
        return tuple(int(value) for value in config.detach().cpu().tolist())

    def _sync(self, model):
        signature = self._signature_for(model)
        if self._signature != signature:
            self.clear()
            self._signature = signature

    def clear(self):
        """Drop cached values and counters while retaining the size limit."""
        self._values.clear()
        self._requests = 0
        self._hits = 0
        self._misses = 0

    def _put(self, key, value):
        self._values[key] = value.detach().clone()
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def seed(self, configs, amplitudes, *, model=None):
        """Insert already-contracted parent amplitudes into the cache."""
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        amplitudes = torch.as_tensor(amplitudes, device=configs.device).reshape(-1)
        if configs.shape[0] != amplitudes.shape[0]:
            raise ValueError("configs and amplitudes must have the same length.")
        if model is not None:
            self._sync(model)
        for config, amplitude in zip(configs, amplitudes):
            self._put(self._key(config), amplitude)

    def evaluate(self, model, configs, *, chunk_size=None):
        """Evaluate amplitudes, contracting only configurations not cached."""
        torch = _require_torch()
        if torch.is_grad_enabled():
            raise RuntimeError("TorchAmplitudeCache is only valid in no-grad mode.")
        configs = _as_long_matrix(configs)
        self._sync(model)
        if configs.shape[0] == 0:
            return torch.empty(0, dtype=torch.get_default_dtype(), device=configs.device)
        unique_configs, inverse = _unique_config_rows(configs)
        if inverse is None:
            inverse = torch.zeros(1, dtype=torch.long, device=configs.device)
        values = [None] * int(unique_configs.shape[0])
        missing = []
        for index, config in enumerate(unique_configs):
            self._requests += 1
            key = self._key(config)
            value = self._values.get(key)
            if value is None:
                missing.append(index)
                self._misses += 1
            else:
                values[index] = value.to(device=configs.device)
                self._hits += 1
                self._values.move_to_end(key)
        if missing:
            missing_index = torch.as_tensor(
                missing, dtype=torch.long, device=configs.device,
            )
            computed = _call_amplitude_fn(
                model, unique_configs[missing_index], chunk_size=chunk_size,
            )
            for offset, index in enumerate(missing):
                value = computed[offset].detach()
                values[index] = value
                self._put(self._key(unique_configs[index]), value)
        return torch.stack(values)[inverse]

    def wrap(self, model):
        """Return a callable model proxy using this cache."""
        return _CachedAmplitudeModel(model, self)

    def snapshot(self):
        """Return lightweight counters suitable for a profile/result JSON."""
        return {
            "entries": int(len(self._values)),
            "max_entries": int(self.max_entries),
            "requests": int(self._requests),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "hit_rate": (
                float(self._hits) / float(self._requests)
                if self._requests else 0.0
            ),
        }


class _CachedAmplitudeModel:
    """Delegate PEPS-specific connected work while caching target rows."""

    def __init__(self, model, cache):
        self._model = model
        self._cache = cache

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, configs, *args, **kwargs):
        chunk_size = kwargs.pop("chunk_size", None)
        if args or kwargs:
            return self._model(configs, *args, **kwargs)
        return self._cache.evaluate(self._model, configs, chunk_size=chunk_size)

    def connected_amplitudes(
        self,
        configs,
        amplitudes,
        connections,
        *,
        chunk_size=None,
        reuse_diagonal=True,
    ):
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        amplitudes = torch.as_tensor(amplitudes, device=configs.device).reshape(-1)
        self._cache._sync(self._model)
        self._cache.seed(configs, amplitudes)
        target_configs = _as_long_matrix(connections.configs)
        if target_configs.shape[0] == 0:
            return torch.empty(0, dtype=amplitudes.dtype, device=configs.device)

        unique_targets, inverse = _unique_config_rows(target_configs)
        if inverse is None:
            inverse = torch.zeros(1, dtype=torch.long, device=configs.device)
        values = [None] * int(unique_targets.shape[0])
        missing = []
        representative = {}
        for index, config in enumerate(target_configs):
            representative.setdefault(self._cache._key(config), index)
        for index, config in enumerate(unique_targets):
            self._cache._requests += 1
            key = self._cache._key(config)
            value = self._cache._values.get(key)
            if value is None:
                missing.append(index)
                self._cache._misses += 1
            else:
                values[index] = value.to(
                    device=configs.device, dtype=amplitudes.dtype,
                )
                self._cache._hits += 1
                self._cache._values.move_to_end(key)

        if missing:
            missing_index = torch.as_tensor(
                missing, dtype=torch.long, device=configs.device,
            )
            representative_ids = torch.as_tensor(
                [representative[self._cache._key(config)] for config in unique_targets[missing_index]],
                dtype=torch.long,
                device=configs.device,
            )
            subset = TorchConnections(
                configs=unique_targets[missing_index],
                coeffs=torch.ones(
                    len(missing), dtype=amplitudes.dtype, device=configs.device,
                ),
                batch_ids=connections.batch_ids[representative_ids],
            )
            connected = getattr(self._model, "connected_amplitudes", None)
            if callable(connected):
                computed = connected(
                    configs,
                    amplitudes,
                    subset,
                    chunk_size=chunk_size,
                    reuse_diagonal=reuse_diagonal,
                )
            else:
                computed = _call_amplitude_fn(
                    self._model, subset.configs, chunk_size=chunk_size,
                )
            computed = torch.as_tensor(computed, device=configs.device).reshape(-1)
            for offset, index in enumerate(missing):
                value = computed[offset].detach()
                values[index] = value
                self._cache._put(self._cache._key(unique_targets[index]), value)
        return torch.stack(values)[inverse].to(dtype=amplitudes.dtype)
