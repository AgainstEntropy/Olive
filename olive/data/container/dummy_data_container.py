# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import ClassVar

from olive.data.constants import DataComponentType, DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER)
class DummyDataContainer(DataContainer):
    """Dummy data container.

    The way to create a dummy data container:
        dummy_data_config = DataConfig(
            name="dummy",
            type="DummyDataContainer",
            load_dataset_config=DataComponentConfig(
                params={
                    "input_names": metric.user_config.input_names,
                    "input_shapes": metric.user_config.input_shapes,
                    "input_types": metric.user_config.input_types,
                }
            ),
            dataloader_config=DataComponentConfig(
                # TODO(shaahji): Remove this override once DataConfig transition is complete
                params={ "batch_size": None }
            )
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "dummy_dataset",
        DataComponentType.DATALOADER.value: "default_dataloader",
    }
