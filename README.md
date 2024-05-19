# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML on on the same dataset. Accuracy of both pipelines is compared in the end.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
We are given a marketing dataset from banking customers. I have created a model which predicts whether or not the customer will subscribe to a term deposit at the bank.

The best performing model was Automated ML -- VotingEnsemble

## Scikit-learn Pipeline
pipeline architecture
Data Preparation -> Feature Engineering -> Hyperparameter Sampling -> Early Stopping Policy -> HyperDrive Configuration
1. Data Preparation -> clean_data() function in train.py file. It takes a tabular dataset as input
2. Feature Engineering -> using train_test_split method to split the cleaned data into two categories i.e. train and test data
3. Hyperparameter Sampling -> Using RandomParameterSampling for Logistic regression. specified range for 'C' (regularization) and 'max_iter' (maximum iterations).
4. Early Stopping Policy -> Used BanditPolicy
5. HyperDrive Configuration -> 
    hyperdrive_config = HyperDriveConfig(
        run_config=src,  # Use the defined ScriptRunConfig object
        primary_metric_name="Accuracy",  # Set the primary metric for evaluation
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,  # Set the goal (maximize/minimize)
        max_concurrent_runs=4,
        max_total_runs=20,  # Set the maximum number of runs
        hyperparameter_sampling=ps,
        policy=policy  # Include policy object if using early termination
    )

I chose RandomParameterSampling as parameter sampler. Benefits : 
-Parallelization : Since each random sample is independent, the evaluation of different hyperparameter combinations can be easily parallelized. This can significantly speed up the hyperparameter tuning process, especially when dealing with large search spaces or computationally expensive models.
-It requires less computation.

I chose BanditPolicy as Early Stopping Policy. Benefits : 
-Reduced Training Time: helped to avoid wasting resources on models that are unlikely to improve further
-Efficient Hyperparameter Tuning: stopping poorly performing runs early, this policy focuses resources on exploring hyperparameter combinations that have a higher chance of yielding better results.

## AutoML
The best-performing model produced by AutoML is a Voting Ensemble. 
In ensemble learning, a Voting Ensemble is a technique that combines the predictions of multiple base learners (models) to create a single, more robust prediction. It leverages the "wisdom of the crowd" principle, aiming to improve model performance by aggregating the outputs of diverse models.

## Pipeline comparison
Scikit-learn : 
hyperparameters : {"C": 1.2137867117323344, "max_iter": 100}
Accuracy - 91.82094%

Auto ML :
Algorithm name - VotingEnsamble
Accuracy - 91.848%

## Future work
Since, for our project's AutoML run, in the automl_config we defined experiment_timeout_minutes=30, the experiment could have given better results if we would have given more time. We can train this model with the new data to keep this model updated.
