import logging
try:
    # Import your logger package to run its configuration code (sets up handlers)
    from src.logger import logger as app_logger
except Exception:
    # Fallback: basic config so we still see logs
    logging.basicConfig(level=logging.INFO)

# Get a logger for this specific main script
log = logging.getLogger(__name__)

# Import all your pipeline components
from src.prediction_pipeline_for_existing_review.data_loading import (
    create_db_engine,
    fetch_pending_reviews,
)
from src.prediction_pipeline_for_existing_review.data_transformation import DataTransformation
from src.prediction_pipeline_for_existing_review.llm_analyzer import LLMAnalysis
from src.prediction_pipeline_for_existing_review.datasaver import DataSaver

def main():
    log.info("--- Starting Prediction Pipeline ---")
    
    # === STAGE 1: DATA LOADING ===
    log.info("Initiating Stage 1: Data Loading")
    db_engine = create_db_engine()
    if not db_engine:
        log.error("Aborting pipeline due to database connection failure."); return

    pending_df = fetch_pending_reviews(engine=db_engine, batch_size=50)
    if pending_df.empty:
        log.info("No pending reviews found. Exiting pipeline."); return
    log.info(f"Stage 1 complete. Fetched {len(pending_df)} reviews.")
    
    # === STAGE 2: DATA TRANSFORMATION ===
    log.info("Initiating Stage 2: Data Transformation")
    try:
        transformer = DataTransformation()
        llm_ready_data = transformer.prepare_data_for_llm(pending_df)
        log.info(f"Stage 2 complete. Transformed {len(llm_ready_data)} records for LLM analysis.")
    except Exception as e:
        log.error(f"Pipeline failed at Stage 2: Data Transformation. Error: {e}"); return

    # === STAGE 3: LLM ANALYSIS ===
    log.info("Initiating Stage 3: LLM Analysis")
    try:
        analyzer = LLMAnalysis(model_name="deepseek-r1:8b")
        final_results = analyzer.run_analysis_on_list(llm_ready_data)
        log.info(f"Stage 3 complete. Successfully analyzed {len(final_results)} records.")
    except Exception as e:
        log.error(f"Pipeline failed at Stage 3: LLM Analysis. Error: {e}"); return

    # === STAGE 4: SAVING RESULTS ===
    log.info("Initiating Stage 4: Saving Results")
    if final_results:
        try:
            saver = DataSaver()
            saver.save_results(final_results)
            log.info("Stage 4 complete. Results saved to database.")
        except Exception as e:
            log.error(f"Pipeline failed at Stage 4: Saving Results. Error: {e}"); return
    else:
        log.warning("Stage 4 skipped: No results from analysis stage to save.")
    
    log.info("--- Prediction Pipeline Finished Successfully ---")
    
if __name__ == "__main__":
    main()