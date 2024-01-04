# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler, PyTorchModelHandler, QNNModelHandler, TensorFlowModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.qnn.utils.local import run_qnn_command

logger = logging.getLogger(__name__)


class QNNModelLibGenerator(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "lib_targets": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Specifies the targets to build the models for. Default: aarch64-android x86_64-linux-clang"
                ),
            ),
            "lib_name": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Specifies the name to use for libraries. Default: uses name in <model.bin> if provided, "
                    " else generic qnn_model.so"
                ),
            ),
        }

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        pass

    def _run_for_config(
        self,
        model: Union[TensorFlowModelHandler, PyTorchModelHandler, ONNXModelHandler],
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> QNNModelHandler:
        main_cmd = "qnn-model-lib-generator"

        # input model path's name without suffix
        input_model_path = Path(model.model_path).resolve()
        input_model_bin = input_model_path.parent / (input_model_path.stem + ".bin")
        if not input_model_bin.exists():
            logger.debug("No model.bin found, using generic qnn_model.so")
            input_model_bin = None

        output_model_path = Path(output_model_path).resolve()

        cmd_list = [
            main_cmd,
            f"-c {model.model_path}",
            f"-b {input_model_bin}" if input_model_bin else "",
            f"-t {config['lib_targets']}" if config.get("lib_targets") else "",
            f"-l {config['lib_name']}" if config.get("lib_name") else "",
            f"-o {output_model_path}",
        ]
        run_qnn_command(
            " ".join(cmd_list),
            dev=True,
        )
        return QNNModelHandler(output_model_path, model_file_format=ModelFileFormat.QNN_LIB)
