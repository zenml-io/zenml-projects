import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict
from zenml import pipeline, step
from zenml.logger import get_logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = get_logger(__name__)

@step
def ingest_and_preprocess_data() -> pd.DataFrame:
    logger.info("Starting data ingestion and preprocessing...")
    data = pd.read_csv("http://data.insideairbnb.com/france/ile-de-france/paris/2023-12-12/data/listings.csv.gz")
    
    features = [
        "id", "accommodates", "bedrooms", "beds", "minimum_nights",
        "number_of_reviews", "availability_365", "review_scores_rating",
        "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value"
    ]
    data = data[features]

    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed['id'] = data_imputed['id'].astype(int)

    logger.info(f"Preprocessed data shape: {data_imputed.shape}")
    return data_imputed

@step
def sample_data(data: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    logger.info(f"Sampling {sample_size} listings from the dataset...")
    return data.sample(n=sample_size, random_state=42)

@step
def analyze_and_train_model(
    data: pd.DataFrame
) -> Tuple[RandomForestRegressor, StandardScaler, List[str]]:
    logger.info("Starting model training...")
    features = [
        "accommodates", "bedrooms", "beds", "minimum_nights",
        "number_of_reviews", "availability_365"
    ]
    X = data[features]
    y = data["review_scores_rating"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importance = sorted(zip(model.feature_importances_, features), reverse=True)
    important_features = [f[1] for f in feature_importance[:3]]
    
    logger.info(f"Model trained. Top 3 important features: {important_features}")
    return model, scaler, important_features

@step
def evaluate_listings(
    data: pd.DataFrame,
    model: RandomForestRegressor,
    scaler: StandardScaler,
    important_features: List[str]
) -> List[Dict]:
    logger.info("Evaluating listings...")
    features = [
        "accommodates", "bedrooms", "beds", "minimum_nights",
        "number_of_reviews", "availability_365"
    ]
    X = data[features]
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    medians = data[features + [f"review_scores_{score_type}" for score_type in 
                               ["accuracy", "cleanliness", "checkin", "communication", "location", "value"]]].median()
    
    improvements = (data[features] < medians[features]).astype(int)
    score_improvements = (data[[f"review_scores_{score_type}" for score_type in 
                                ["accuracy", "cleanliness", "checkin", "communication", "location", "value"]]] 
                          < medians[[f"review_scores_{score_type}" for score_type in 
                                     ["accuracy", "cleanliness", "checkin", "communication", "location", "value"]]]).astype(int)
    
    all_improvements = pd.concat([improvements, score_improvements], axis=1)
    
    evaluations = []
    for i, row in enumerate(data.itertuples()):
        eval_dict = {
            "id": row.id,
            "predicted_score": predictions[i],
            "actual_score": row.review_scores_rating,
            "areas_for_improvement": all_improvements.columns[all_improvements.iloc[i] == 1].tolist()
        }
        evaluations.append(eval_dict)
    
    logger.info(f"Evaluated {len(evaluations)} listings")
    return evaluations

@step(enable_cache=False)
def generate_recommendations(evaluations: List[Dict]) -> List[str]:
    logger.info("Generating recommendations using LLM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)

    recommendations = []
    for i, eval in enumerate(evaluations):
        prompt = f"As an Airbnb listing optimization expert, provide three specific, actionable recommendations to improve this Airbnb listing based on the following evaluation:\n"
        prompt += f"Predicted review score: {eval['predicted_score']:.2f}\n"
        prompt += f"Actual review score: {eval['actual_score']:.2f}\n"
        prompt += f"Areas for improvement: {', '.join(eval['areas_for_improvement'])}\n"
        prompt += "Your recommendations should be practical, specific, and tailored to the areas of improvement identified."

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.95)
        recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        recommendations.append(recommendation)
        
        if i < 5:  # Print first 5 recommendations
            logger.info(f"Sample recommendation for listing {eval['id']}:\n{recommendation}\n")

    logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations

@step
def generate_reports(
    data: pd.DataFrame,
    evaluations: List[Dict],
    recommendations: List[str]
) -> List[str]:
    logger.info("Generating final reports...")
    reports = []
    for eval, rec in zip(evaluations, recommendations):
        listing = data[data["id"] == eval["id"]].iloc[0]
        report = f"Report for Listing {eval['id']}:\n\n"
        report += f"Current Overall Rating: {eval['actual_score']:.2f}\n"
        report += f"Predicted Rating: {eval['predicted_score']:.2f}\n\n"
        report += "Detailed Scores:\n"
        for score_type in ["accuracy", "cleanliness", "checkin", "communication", "location", "value"]:
            score_col = f"review_scores_{score_type}"
            report += f"- {score_type.capitalize()}: {listing[score_col]:.2f}\n"
        report += f"\nAreas for Improvement: {', '.join(eval['areas_for_improvement'])}\n\n"
        report += f"Recommendations:\n{rec}\n"
        reports.append(report)
    
    logger.info(f"Generated {len(reports)} reports")
    
    # Print a few sample reports
    for i in range(min(3, len(reports))):
        logger.info(f"Sample Report {i+1}:\n{reports[i]}\n{'='*50}\n")
    
    return reports

@pipeline
def airbnb_advisor_pipeline():
    data = ingest_and_preprocess_data()
    # data = sample_data(data)
    model, scaler, important_features = analyze_and_train_model(data)
    evaluations = evaluate_listings(data, model, scaler, important_features)
    recommendations = generate_recommendations(evaluations)
    reports = generate_reports(data, evaluations, recommendations)
    return reports

if __name__ == "__main__":
    airbnb_advisor_pipeline()