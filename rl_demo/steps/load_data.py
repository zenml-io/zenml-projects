"""Register and version training dataset with lineage metadata."""

from typing import Annotated

from steps.models import DatasetMetadata
from zenml import log_metadata, step


@step
def load_training_data(
    env_names: list[str],
    client_id: str = "acme-corp",
    project: str = "rl-optimization",
    data_source: str = "internal-simulation",
    domain: str = "operations-research",
) -> Annotated[DatasetMetadata, "training_dataset"]:
    """
    Register and version the training dataset with lineage metadata.

    This step creates a traceable artifact at the root of every training run.
    In the ZenML dashboard, you can click on any model version and trace back
    to this exact dataset — including which client, project, and data source
    produced it.

    For data scrubbing / GDPR compliance:
        zenml artifact list --tag client_id:acme-corp
        → shows every artifact (datasets, checkpoints, models) linked to that client
        → trace forward to find all model versions that need to be retrained or deleted
    """
    metadata = DatasetMetadata(
        client_id=client_id,
        project=project,
        data_source=data_source,
        domain=domain,
        env_names=env_names,
        description=f"Training data for {len(env_names)} environments, "
        f"client={client_id}, project={project}",
    )

    log_metadata(
        metadata={
            "client_id": client_id,
            "project": project,
            "data_source": data_source,
            "domain": domain,
            "environments": env_names,
            "compliance": {
                "data_retention_policy": "90d",
                "scrub_eligible": True,
                "lineage_tracked": True,
            },
        },
        infer_artifact=True,
    )

    print(
        f"📦 Registered training dataset: client={client_id}, project={project}"
    )
    print(f"   Data source: {data_source} | Domain: {domain}")
    print(f"   Environments: {env_names}")
    print(
        f"   → Full lineage tracked — traceable for data scrubbing/compliance"
    )

    return metadata
