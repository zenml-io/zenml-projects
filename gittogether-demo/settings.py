import os

from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors import (
    KubernetesOrchestratorSettings,
)

MNT_PATH = "/mnt/data"

docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    environment={
        "PJRT_DEVICE": "CUDA",
        "USE_TORCH_XLA": "false",
        "MKL_SERVICE_FORCE_INTEL": 1,
        "HF_TOKEN": "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",
        # "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": MNT_PATH,
        "ZENML_DISABLE_CLIENT_SERVER_MISMATCH_WARNING": "1",
    },
    python_package_installer="uv",
    requirements="requirements.txt",
    python_package_installer_args={
        "system": None,
    },
    apt_packages=["git", "ffmpeg"],
    prevent_build_reuse=True,
)

sd_docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    environment={
        "PJRT_DEVICE": "CUDA",
        "USE_TORCH_XLA": "false",
        "MKL_SERVICE_FORCE_INTEL": 1,
        "HF_TOKEN": "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",
        # "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": MNT_PATH,
        "ZENML_DISABLE_CLIENT_SERVER_MISMATCH_WARNING": "1",
    },
    python_package_installer="uv",
    requirements="requirements-sd.txt",
    python_package_installer_args={
        "system": None,
    },
    apt_packages=["git", "ffmpeg"],
    prevent_build_reuse=True,
)

flux_docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    environment={
        "PJRT_DEVICE": "CUDA",
        "USE_TORCH_XLA": "false",
        "MKL_SERVICE_FORCE_INTEL": 1,
        "HF_TOKEN": "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",
        # "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": MNT_PATH,
        "ZENML_DISABLE_CLIENT_SERVER_MISMATCH_WARNING": "1",
    },
    python_package_installer="uv",
    requirements="requirements-flux.txt",
    python_package_installer_args={
        "system": None,
    },
    apt_packages=["git", "ffmpeg"],
    prevent_build_reuse=True,
)


kubernetes_settings = KubernetesOrchestratorSettings(
    pod_settings={
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "zenml.io/gpu",
                                    "operator": "In",
                                    "values": ["yes"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "volumes": [
            {
                "name": "data-volume",
                "persistentVolumeClaim": {"claimName": "pvc-managed-premium"},
            }
        ],
        "volume_mounts": [{"name": "data-volume", "mountPath": MNT_PATH}],
    },
)
