.. customcarditem::
   :header: Quantile
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/graph_classification.svg
   :tags: Aggregation

.. include:: ../links.rst

########
Quantile
########

Module Interface
________________

.. autoclass:: torchmetrics.aggregation.QuantileMetric
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.aggregation.quantile

.. autofunction:: torchmetrics.functional.aggregation.quantile_rank_error_bound
