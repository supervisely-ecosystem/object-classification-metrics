# Overview
This app is an interactive visualization for classification metrics. It allows to estimate performance of a model by comparing ground truth annotations with predictions. The app supports single-label and multi-label tasks.

# How to use
1. Select two projects: with ground truth labels and with model predictions
2. The app matches images in both projects and produces detailed report about matched pairs, allowing to validate the data
3. Then you can match classes and select that will be used in metrics calculation. If your labels have a suffix (like "_nn") that was auto-generated after the [Apply Classifier App](https://ecosystem.supervise.ly/apps/apply-classification-model-to-project), you can make sure that they are matched too.
4. Click "Calculate" and you will see a confusion matrix. Click on the cells to see the images and where the model gets wrong. You can preview images from the table.
5. Check the tabs "Overall" and "Per class" to see the metrics like precision, recall, f1-score and others.
