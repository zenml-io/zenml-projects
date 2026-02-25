"""Promote the best policy via ZenML's Model Control Plane."""

from typing import Annotated

from materializers.policy_checkpoint_materializer import (
    PolicyCheckpointMaterializer,
)
from steps.models import EvalResult, PolicyCheckpoint
from zenml import ArtifactConfig, log_metadata, step
from zenml.client import Client
from zenml.enums import ArtifactType, ModelStages


@step(
    output_materializers={
        "promoted_policy_checkpoint": PolicyCheckpointMaterializer
    },
)
def promote_best_policy(
    eval_results: list[EvalResult],
    policy_checkpoints: list[PolicyCheckpoint],
) -> Annotated[
    PolicyCheckpoint,
    ArtifactConfig(
        name="promoted_policy_checkpoint", artifact_type=ArtifactType.MODEL
    ),
]:
    """
    Promote the best policy via ZenML's Model Control Plane.

    Returns the winning checkpoint as a model artifact — load via
    model_version.get_artifact("promoted_policy_checkpoint").load().
    """
    assert len(eval_results) == len(policy_checkpoints), (
        "eval_results and checkpoints must be parallel"
    )
    tag_to_checkpoint = {
        r.tag: c for r, c in zip(eval_results, policy_checkpoints)
    }

    winner = max(eval_results, key=lambda r: r.eval_mean_reward)
    promoted = tag_to_checkpoint[winner.tag]

    print(
        f"🏆 Promoting {winner.tag} → "
        f"reward {winner.eval_mean_reward:.2f} ± {winner.eval_std_reward:.2f}"
    )

    client = Client()
    latest_version = client.get_model_version(model_name_or_id="rl_policy")
    client.update_model_version(
        model_name_or_id="rl_policy",
        version_name_or_id=latest_version.name,
        stage=ModelStages.PRODUCTION,
        force=True,
    )
    print(f"📋 Model version '{latest_version.name}' → stage: production")

    log_metadata(
        metadata={
            "promoted_tag": winner.tag,
            "promoted_reward": float(winner.eval_mean_reward),
            "stage_transition": "staging → production",
            "promotion_criteria": "highest eval reward",
        },
        infer_model=True,
    )
    return promoted
