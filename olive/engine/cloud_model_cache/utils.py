# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import io
import json
import logging
from copy import deepcopy
from pathlib import Path

from olive.common.utils import hash_dict
from olive.model.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class OliveCloudCacheHelper:
    def __init__(self, passes_hash: str, cache_dir: str, input_model_config: ModelConfig):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError:
            raise ImportError(
                "Please install azure-storage-blob and azure-identity to use the cloud model cache feature."
            )
        account_url = "https://olivepublicmodels.blob.core.windows.net"
        credential = self._get_credentials()
        self.container_name = "olivecachemodels"
        self.container_client = ContainerClient(
            account_url=account_url, container_name=self.container_name, credential=credential
        )

        self.map_blob = "model_map.json"

        self.local_map_path = Path(cache_dir) / "model_map.json"
        self.metadata_path = Path(cache_dir) / "metadata.json"
        self.local_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_blob = passes_hash
        self.metadata = f"{self.model_blob}/metadata.json"

        self.input_model_config = input_model_config
        self.cache_dir = cache_dir

    def get_metadata(self):
        logger.info(f"Downloading model cache map from {self.metadata}")
        blob_client = self.container_client.get_blob_client(self.metadata)

        if not blob_client.exists():
            logger.error(f"Blob {self.map_blob} does not exist in container {self.container_name}.")
            return ([], None)

        self.metadata_path.unlink(missing_ok=True)
        self.metadata_path.touch()
        with open(self.metadata_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        with open(self.metadata_path) as file:
            map_dict = json.load(file)
        return map_dict["model_ids"], map_dict["pass_id"]

    def update_metadata(self, model_ids, pass_id):
        logger.info("Updating metadata.")
        model_ids = [model_id.split("-")[0].split("_")[1] for model_id in model_ids]
        metadata = {"model_ids": model_ids, "pass_id": pass_id}
        metadata_bytes = json.dumps(metadata).encode()
        with io.BytesIO(metadata_bytes) as data:
            self.container_client.upload_blob(self.metadata, data=data, overwrite=True)

    def download_cached_model_from_blob(self, model_path, output_model_path):
        model_directory_prefix = self.model_blob
        blob_list = self.container_client.list_blobs(name_starts_with=model_directory_prefix)
        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)

        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)
            local_file_path = output_model_path / blob.name[len(model_directory_prefix) + 1 :]
            logger.info(f"Downloading {blob.name} to {local_file_path}")
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

        # model config
        with open(output_model_path / "model_config.json") as file:
            model_config_dict = json.load(file)
            model_config = ModelConfig.from_json(model_config_dict)

            model_config.config["model_path"] = str(output_model_path / "model" / model_path)
        return model_config

    def download_model_cache_map(self):
        logger.info(f"Downloading model cache map from {self.map_blob}")
        blob_client = self.container_client.get_blob_client(self.map_blob)

        if not blob_client.exists():
            logger.warning(f"Blob {self.map_blob} does not exist in container {self.container_name}.")
            return None

        self.local_map_path.unlink(missing_ok=True)
        self.local_map_path.touch()
        with open(self.local_map_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        with open(self.local_map_path) as file:
            map_dict = json.load(file)
        return map_dict

    def exist_in_model_cache_map(self, cloud_cache_map: dict):
        model_path = cloud_cache_map.get(self.model_blob, None)
        if not model_path:
            logger.warning("Model cache not found in the cloud.")
            return None
        return model_path

    def update_model_cache_map(self, cloud_cache_map, blob):
        logger.info("Updating model cache map.")
        if not self.local_map_path.exists():
            self.local_map_path.touch()
            with open(self.local_map_path, "w") as file:
                file.write("{}")
        with open(self.local_map_path, "r+b") as file:
            cloud_cache_map = json.load(file)
        cloud_cache_map[self.model_blob] = blob

        cache_map_bytes = json.dumps(cloud_cache_map).encode()
        with io.BytesIO(cache_map_bytes) as data:
            self.container_client.upload_blob(self.map_blob, data=data, overwrite=True)

    def upload_model_to_cloud_cache(self, output_model_config):
        logger.info("Uploading model to cloud cache.")
        model_path = output_model_config.config.get("model_path")

        if model_path is None:
            logger.error("Model path is not found in the output model config.")
            return "INVALID"

        model_path = Path(model_path)
        model_config_copy = deepcopy(output_model_config)
        model_config_copy.config["model_path"] = None

        # upload model config file
        model_config_bytes = json.dumps(model_config_copy.to_json()).encode()
        with io.BytesIO(model_config_bytes) as data:
            self.container_client.upload_blob(f"{self.model_blob}/model_config.json", data=data, overwrite=True)

        # upload model file
        model_blob = str(Path(self.model_blob) / "model")
        if not model_path.is_dir():
            with open(model_path, "rb") as data:
                self.container_client.upload_blob(f"{model_blob}/{model_path.name}", data=data, overwrite=True)
        else:
            self._upload_dir_to_blob(model_path, f"{model_blob}")
        return model_path.name

    def _upload_dir_to_blob(self, dir_path, blob_folder_name):
        for item in dir_path.iterdir():
            if item.is_dir():
                self._upload_dir_to_blob(item, f"{blob_folder_name}/{item.name}")
            else:
                blob_name = f"{blob_folder_name}/{item.name}"
                with open(item, "rb") as data:
                    self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)

    def _get_credentials(self):
        """Get credentials for MLClient.

        Order of credential providers:
        1. Azure CLI
        2. DefaultAzureCredential
        3. InteractiveBrowserCredential
        """
        from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

        logger.debug("Getting credentials for MLClient")
        try:
            default_auth_params = {"exclude_managed_identity_credential": True}
            credential = DefaultAzureCredential(**default_auth_params)
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
            logger.debug("Using DefaultAzureCredential")
        except Exception:
            logger.warning("Using InteractiveBrowserCredential since of default credential errors", exc_info=True)
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = InteractiveBrowserCredential()

        return credential