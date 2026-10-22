# Observations

## 1. How does Online Inference differ from Batch Inference?
Online inference predicts an individual record when a request arrives and returns the result immediately. Batch inference processes a group of records together in one operation. Online inference is useful for individual, low-latency requests, while batch inference is efficient for many records that can be processed together.

## 2. What is the difference between Streaming Inference and Batch Inference?
Streaming inference handles incoming records continuously or one at a time as they arrive, so each record can receive a prediction with low delay. Batch inference waits until a group of records is available and processes that group together. In this assignment, streaming is simulated by processing 10 held-out records sequentially with a one-second delay.

## 3. Why is FastAPI used for model serving?
FastAPI exposes the trained model through HTTP endpoints, allowing other applications to request predictions without directly loading the model. It supports request validation with Pydantic, generates interactive Swagger documentation at `/docs`, and can return structured JSON responses.

## 4. Why should the same trained model be reused for all inference strategies?
Reusing the same saved pipeline keeps the preprocessing and model consistent across online, batch, streaming, and API predictions. It avoids differences caused by retraining or using different preprocessing steps and makes results easier to compare and reproduce.

## Results to fill in after running
- Test accuracy: copy the value from `training_metrics.json`.
- Online prediction: copy the prediction printed by `online_inference.py`.
- Batch inference: confirm that `batch_predictions.csv` contains 20 rows.
- Streaming inference: confirm that 10 records were processed.
- API checks: record the successful `/health` and `/predict` responses and the invalid request's HTTP 422 validation response.
