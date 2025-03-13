zenml model delete ecb_interest_rate_model 

zenml stack set local-gcp-step-operator 

python run.py --etl --mode develop

python run.py --feature --mode develop

python run.py --training --mode develop