# Architecture notes

## Request and experiment flow

The API is a thin interface over modular ingestion, preprocessing, training, evaluation, registry, and prediction services. Dataset adapters describe source columns and target semantics. Fitted transformations and resampling remain within training folds; final test data remains untouched until evaluation. For time-aware fraud data, split chronologically when a trustworthy timestamp exists.

The optional hybrid path computes unsupervised features such as anomaly scores from training partitions and appends them to supervised model inputs. These features are not labels. Clusters and anomaly flags are presented as patterns for investigation, not evidence that a transaction is fraudulent.

## Persistence boundaries

PostgreSQL stores users, dataset metadata, experiment metadata, model versions, metrics, predictions, audit events, and (where configured) application/transaction records. Raw financial datasets and serialized model artifacts remain in dedicated local or object storage paths, with sensitive data excluded from source control.

## Initial environment

`docker-compose.yml` currently starts PostgreSQL for local development. API and frontend can be run independently for faster iteration. The API health route reports process readiness; database readiness checks and migrations will be added with persistence work.
