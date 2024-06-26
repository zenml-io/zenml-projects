import wandb
from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset

# Initialize wandb
wandb.init(project="zenml_embeddings", entity="strickvl")

model = SentenceTransformer("embedding-data/distilroberta-base-sentence-transformer")

dataset_name = "zenml/rag_qa_embedding_questions"
train_dataset = load_dataset(dataset_name, split="train")
test_dataset = load_dataset(dataset_name, split="test")

train_examples = []
train_data = train_dataset
n_examples = train_dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    train_examples.append(
        InputExample(texts=[example["generated_questions"][0], example["page_content"]])
    )

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 30
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

# Configure wandb
wandb.config.update(
    {"num_epochs": num_epochs, "warmup_steps": warmup_steps, "batch_size": 32}
)

# Track the training loss
wandb.watch(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
)

# Evaluation
test_examples = []
test_data = test_dataset
n_test_examples = test_dataset.num_rows

for i in range(n_test_examples):
    example = test_data[i]
    test_examples.append(
        InputExample(texts=[example["generated_questions"][0], example["page_content"]])
    )


class InputExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.texts


test_dataset = InputExampleDataset(test_examples)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

total_similarity = 0
for batch in test_dataloader:
    question_embeddings = model.encode(batch[0])
    content_embeddings = model.encode(batch[1])
    similarity_scores = cosine_similarity(question_embeddings, content_embeddings)
    total_similarity += similarity_scores.diagonal().sum()

average_similarity = total_similarity / n_test_examples
print(f"Average cosine similarity on the test set: {average_similarity}")

# Log the average cosine similarity to wandb
wandb.log({"avg_cosine_similarity": average_similarity})

# Finish the wandb run
wandb.finish()
