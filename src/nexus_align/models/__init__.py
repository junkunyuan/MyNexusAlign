"""Model registry: register built-in model factories by name on import."""

from nexus_align.registry import registry
from nexus_align.models.meanflow_sit import MeanFlowSiT_models, MeanFlowSiTModel

# All sizes share MeanFlowSiTModel; it picks the model from cfg.model.name.
for _name in MeanFlowSiT_models:
    registry.register("model", _name, MeanFlowSiTModel)
