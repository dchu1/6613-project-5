# 6613-project-5
Author: David Chu

Net ID: dfc296

# Part 1
Moved relevent code from [covid-cxr repository](https://github.com/aildnont/covid-cxr) into colab notebook. Then trained the model, and generated a lime explanation for the first image in the test set.

Plase note that if you want to run the colab notebook, you need to make sure to do 2 things:

1. After installing the dependencies, restart the runtime. This is because colab notebooks get preloaded with matplotlib, mpl_toolkits, numpy, pandas, tqdm, which are replaced with older versions. If the runtime is not restarted, it will not use the older versions.

2. Import the kaggle dataset manually (either upload it or load from google drive)

## Experiment:
1. Trained the model for **Binary Classification** for covid-19 using the 31 layer **DCNN resnet**.
2. Created LIME explanation for first image in Test Set.

## Results:
| Metric          | Value             |
|-----------------|-------------------|
| loss            | 342.5124877929687 |
| accuracy        | 0.89312977        |
| precision       | 0.8125            |
| recall          | 0.5416667         |
| auc             | 0.9427772         |
| f1score         | 0.65              |
| True Positives  | 104               |
| False Positives | 3                 |
| False Negatives | 11                |
| True Negatives  | 13                |

![Confusion Matrix](https://i.imgur.com/ivI4QLR.png)

![LIME Explanation](https://i.imgur.com/sUmzEPk.png)

## Conclusion:
The metrics are slightly worse than what was in the associated [post](https://towardsdatascience.com/investigation-of-explainable-predictions-of-covid-19-infection-from-chest-x-rays-with-machine-cb370f46af1d) by the authors. This could be due to several reasons:

1. The default configuration has the training end after validation loss does not decrease for 7 epochs. As such, our model only ran for 11/200 epochs before stopping. It is possible that we could get better results if we increased this threshold, but due to compute resource limitations I did not want to increase the threshold value.

2. Using resnet instead of the dcnn. Again, due to compute limitations I used the default dcnn resnet, but it is possible that using the resnet50 or resnet101 might yield better results.

Looking at the LIME explanation, you can see that both the negative and positive indicators mostly lie outside the lungs. This implies that model is not using the right features to make its classification. This is inline with the author's obervations as well.

# Part 2
See attached PDF for explanation of SHAP

# Part 3
I modified the SHAP GradientExplainer [example](https://github.com/slundberg/shap/blob/master/notebooks/gradient_explainer/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet.ipynb) to use our model, and explain the 17th layer for the first image in the test set

## Results
![SHAP Explanation](https://i.imgur.com/tHg24Kz.png)

*note that since it was run on a binary classifier, the non-COVID-19 and COVID-19 explanations are mirror images of each other*

## Conclusion
Similar to the LIME explanation, most of the features that most influenced the prediction lie outside the lungs. This is at least encouraging in the sense that the two explanations are not wildly diverging.

## Difficulties faced
1. This specific example was made 2 years ago, and uses a much older version of Tensorflow and Keras. As a result, it was necessary to make several changes:
    - disable tf2 features for compatability
    - import tensorflow.compat.v1.keras.backend instead of the notebooks current version of keras.backend
    
    *note that there is also an [example](https://github.com/slundberg/shap/blob/master/notebooks/gradient_explainer/Multi-input%20Gradient%20Explainer%20MNIST%20Example.ipynb) that is compatible with TF 2.0, but uses a much simpler 1 CNN block model*

2. As part of preprocessing the images, the pixels are standarized (pixel values scaled to have a zero mean and unit variance):

    `test_img_gen = ImageDataGenerator(preprocessing_function=remove_text, samplewise_std_normalization=True, samplewise_center=True)`

    As a result, I had to modify the shap.image_plot code to draw the original image rather than the preprocessed image.

