"""Exact site-layer preparation for Quimb's sequential 2D boundaries."""


def prepare_quimb_site_layers(network, layer_tags):
    """Return an owned network when a site/layer contains multiple factors.

    Quimb's sequential boundary contractions expect one tensor per incoming
    site/layer. Split gate targets can have several; contract only those
    groups exactly, preserving the requested layer order and original target.
    Ordinary two-layer PEPS networks need no copy or extra contractions.
    """
    if layer_tags is None:
        return network
    tag_map = getattr(network, 'tag_map', {})
    layers = {layer: set(tag_map.get(layer, ())) for layer in layer_tags}
    groups = [
        (site, layer)
        for site in getattr(network, 'site_tags', ())
        for layer, tids in layers.items()
        if len(set(tag_map.get(site, ())) & tids) > 1
    ]
    if not groups:
        return network
    prepared = network.copy()
    for tags in groups:
        prepared.contract_(tags, which='all', optimize='greedy')
    return prepared
