# cooking-activity-detection

This code explores the task of cooking activity detection using a small amount of data obtained from YouTube and a simple transfer-learned network using models from the Epic Kitchens repository (https://github.com/epic-kitchens). 

To run the code, it is recommended you create a conda environment using the included `environment.yml` file. If this doesn't work, please see additional instructions from the Epic Kitchens repository on environment setup for running their models: https://github.com/epic-kitchens/epic-kitchens-55-action-models

Once the environment is activated, the code can be run by simply executing: 

`python cooking_activity_detection.py`

This will load the training and test data, extract the features, train the model, and produce results in the form of a test video, `labeled.mp4`, a confusion matrix, and some simple performance metrics.

The report, `cooking_activity_detection.pdf`, includes some insights and discussion on this exercise.

