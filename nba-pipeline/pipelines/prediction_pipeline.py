from zenml.pipelines import pipeline


@pipeline
def inference_pipeline(
    importer,
    preprocessor,
    extract_next_week,
    model_picker,
    predictor,
    post_processor,
    prediction_poster,
):
    """Defines an inference pipeline to get prediction results of next week's
    NBA games.

    Args:
        importer: Import step to query data.
                preprocessor: Preprocess data for inference.
                extract_next_week: Extract next week's result.
                model_picker: Pick the best model from history.
                predictor: Predict the results for next week.
                post_processor: Post-process data for human readability.
                prediction_poster: Post results on Discord.
    """
    season_schedule = importer()
    processed_season_schedule, le_seasons = preprocessor(season_schedule)
    upcoming_week = extract_next_week(processed_season_schedule)
    model, run_id = model_picker()
    predictions = predictor(model, upcoming_week, le_seasons)
    readable_predictions = post_processor(predictions)
    prediction_poster(readable_predictions)
