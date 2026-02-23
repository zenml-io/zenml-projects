"""Generate sweep configuration at runtime."""

from typing import Annotated, Tuple

from zenml import log_metadata, step
from zenml.types import HTMLString

from steps.models import DatasetMetadata, EnvConfig


@step
def configure_sweep(
    dataset: DatasetMetadata,
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
) -> Tuple[
    Annotated[list[EnvConfig], "sweep_configs"],
    Annotated[HTMLString, "sweep_summary"],
]:
    """
    Generate the sweep configuration at runtime.

    This is the "dynamic" part — the number of downstream training steps
    is determined HERE, not hardcoded in the pipeline definition.
    You could also pull this from a config file, database, or API.
    """
    configs = []
    for env_name in dataset.env_names:
        for lr in learning_rates:
            tag = f"{env_name}_lr{lr}"
            configs.append(
                EnvConfig(
                    env_name=env_name,
                    learning_rate=lr,
                    total_timesteps=total_timesteps,
                    device=device,
                    tag=tag,
                )
            )

    log_metadata(
        metadata={
            "sweep_size": len(configs),
            "environments": dataset.env_names,
            "learning_rates": learning_rates,
            "client_id": dataset.client_id,
            "project": dataset.project,
        },
        infer_artifact=True,
    )

    print(f"🐡 Configured sweep: {len(configs)} training runs")
    for c in configs:
        print(f"   → {c.tag}")

    # HTML summary for ZenML dashboard
    rows = "".join(
        f"<tr><td>{c.env_name}</td><td>{c.learning_rate}</td><td>{c.tag}</td></tr>"
        for c in configs
    )
    summary_html = HTMLString(f"""
    <div style="font-family: system-ui, sans-serif; padding: 1rem;">
        <h3>Sweep Configuration</h3>
        <p><b>{len(configs)}</b> training runs • {len(dataset.env_names)} envs × {len(learning_rates)} learning rates</p>
        <table style="border-collapse: collapse;">
            <thead>
                <tr style="background: #eee;">
                    <th style="padding: 0.5rem; text-align: left;">Environment</th>
                    <th style="padding: 0.5rem; text-align: left;">Learning Rate</th>
                    <th style="padding: 0.5rem; text-align: left;">Tag</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """)
    return configs, summary_html
