"""Model registry: register built-in model factories by name on import."""

from nexus_align.registry import registry
from nexus_align.models.meanflow_sit import MeanFlowSiT_models

for _name, _factory in MeanFlowSiT_models.items():
    registry.register("model", _name, _factory)
