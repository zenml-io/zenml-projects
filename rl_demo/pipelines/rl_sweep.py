"""
Dynamic RL training pipeline with PufferLib.

Key properties:
1. DATA LINEAGE: Every run traces back to a versioned dataset artifact
   with client/project metadata — enabling GDPR-style data scrubbing.
2. DYNAMIC: The number of training steps is determined at runtime
   by configure_sweep — not hardcoded in the pipeline definition.
3. FAN-OUT: train_agent.map() creates one isolated step per config,
   each with its own container, GPU, artifacts, and retry logic.
4. FAN-IN: evaluate_agents receives ALL results and compares them.
5. MODEL CONTROL PLANE: Policies are versioned and promoted through
   stages (staging → production) with full lineage in the dashboard.
"""

from zenml import Model, pipeline
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
    KubernetesOrchestratorSettings,
)

from steps import (
    configure_sweep,
    create_sweep_report,
    evaluate_agents,
    load_training_data,
    promote_best_policy,
    train_agent,
)

docker_settings = DockerSettings(
    dockerfile="Dockerfile",
    python_package_installer="pip",
)

kubernetes_settings = KubernetesOrchestratorSettings(
    pod_settings={
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "8Gi",
                "ephemeral-storage": "20Gi",
            },
            "limits": {
                "cpu": "4",
                "memory": "16Gi",
                "ephemeral-storage": "30Gi",
            },
        },
    },
)


@pipeline(
    dynamic=True,
    enable_cache=True,
    model=Model(
        name="rl_policy",
        license="MIT",
        description="PufferLib RL agents across multiple environments",
    ),
    settings={"docker": docker_settings, "orchestrator": kubernetes_settings},
)
def rl_environment_sweep(
    env_names: list[str],
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
    client_id: str = "acme-corp",
    project: str = "rl-optimization",
    data_source: str = "internal-simulation",
    domain: str = "operations-research",
):
    """Dynamic RL training pipeline with PufferLib."""
    dataset = load_training_data(
        env_names=env_names,
        client_id=client_id,
        project=project,
        data_source=data_source,
        domain=domain,
    )

    configs, _ = configure_sweep(
        dataset=dataset,
        learning_rates=learning_rates,
        total_timesteps=total_timesteps,
        device=device,
    )

    train_outputs = train_agent.map(configs)
    training_results, policy_checkpoints, _ = train_outputs.unpack()

    eval_results, _ = evaluate_agents(
        training_results=training_results,
        policy_checkpoints=policy_checkpoints,
    )

    create_sweep_report(training_results=training_results, eval_results=eval_results)

    promote_best_policy(
        eval_results=eval_results,
        policy_checkpoints=policy_checkpoints,
    )
