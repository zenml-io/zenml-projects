from zenml.repository import Repository

repo = Repository()
p = repo.get_pipeline("training_pipeline")
last_run = p.runs[-1]
trainer_step = last_run.get_step("model_trainer")
model = trainer_step.output
