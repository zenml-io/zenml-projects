include .env
export $(shell sed 's/=.*//' .env)

build:
	docker build -t localzen --target base .

run:
	docker run --gpus all -p 8237:8237 -p 8080:8080 -p 8000:8000 --network default --rm -it  -v .:/src -e WANDB_API_KEY=${WANDB_API_KEY} -e HUGGINGFACE_USERNAME=${HUGGINGFACE_USERNAME} -e  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} localzen 

shell:
	docker exec -it localzen bash