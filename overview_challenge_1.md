# Overview

This Kaggle competition is designed as a practical assessment for the *Deep Learning course*. It aims to evaluate your understanding and application of deep learning techniques for image classification. The competition provides hands-on experience in building, training, and evaluating deep learning models on a real-world image dataset.

You will be provided with a dataset with pictures of various celebrities, and your goal is two-fold. First, there are some binary attributes to predict (such as "Smiling", "No_Beard"…). Then, your model must learn to distinguish between 500 unique celebrities, with an additional "other" class for unknown faces.

A notebook template is provided via Moodle, showing an example of how to load the data and prepare the submission.

**RULES (Read Carefully):**
- This is an individual competition, everyone should participate!
- You will need to design a working CNN architecture (using Convolutional Layers), along with other best practices (hyperparameters validation, early stopping, …) and methods of deep learning (regularization, dropout, …).
- The model architecture should be **explicitly** coded, and trained **from scratch in the notebook you use for submission**.
- It is **NOT** allowed to use pre-trained models, or models for image classification not based on CNNs. It is however encouraged to research for solutions, discuss among participants, and use (thoughtfully!) LLMs.
- It is **NOT** allowed to use external datasets, or any data other than the challenge dataset. It is encouraged to use data augmentation techniques on the dataset if you wish to have more data.
- Given the above points, it is allowed to import established external libraries that may be useful. It is highly recommended to use PyTorch, one of the most popular Deep Learning frameworks.
- This competition will be conducted **exclusively** on Kaggle, leveraging its Notebook and evaluation platform. Therefore, only submissions delivered through Kaggle notebooks will be considered, as the submitted notebooks will be inspected to ensure compliance with challenge rules. As a consequence, also **model training needs to be performed inside the notebook**: the purpose of the challenge is to have you use your knowledge gained from the course to tackle deep learning problems, using your skills and reasoning. The usage of a Kaggle notebook with limited compute (30 hours per week of GPU usage) is to ensure a **fair challenge to all students**, otherwise the use of external computational resources would grant an advantage to students that have access to more of them, and would make your results impossible to reproduce by the Teaching Assistants. All notebooks will be delivered for the final evaluation and manually verified by the Teaching Assistants.


# Evaluation

Submissions are evaluated by the average of two scores, one for the attribute predictions, and one for the celebrity multi-class classification. The score for the attribute prediction is the average F1 score across all binary attributes to be predicted. The score for the celebrity classification is macro F1 score, that is the average of F1 score for each of the 501 classes (500 celebrities + 1 "other").


# Submission File

**Refer to the notebook in Moodle for how to properly prepare your Submissions for evaluation**. To prepare your submission correctly for score evaluation, it needs to output a submission.csv file, with the same structure (columns) of train_data.csv.

This file will be processed by Kaggle, and the outcome will be published in the Leaderboard.

There is a **limit of 3 daily submissions**. Plan your evaluations accordingly, rather than following a trial-and-error approach. Start with a few samples to design your pipeline, when everything works train the model on the whole dataset. To speed up computations, it is advised to move the model and data to a GPU (Notebook > Settings > Accelerators). Keep in mind that model training and evaluation are resources and time-consuming, and Kaggle has limits, so plan ahead of the challenge deadline!


# Scoring

Your model will be evaluated on a test, public leaderboard throughout the challenge. After the deadline, all the notebooks will be evaluated on a separate private test set, giving the final ranking. This is meant to **avoid overfitting** on the test set. Your model should perform well and generalize to new data!

If you try several submissions, ensure to flag appropriately the one you want to be considered for the final evaluation. In the case of ties and equal scores, earlier submissions have priority.


# Delivery

As Kaggle notebooks remain private by default, we are unable to access their full content and can only see the title and submission data. To verify that the notebook used for your final submission complies with the challenge rules, at the end of the challenge we ask you to **share the final notebook** you flagged for evaluation with us on the Kaggle platform. To do this:

Open the Kaggle notebook you used for your final submission (we can see the specific version used for the submission).
Click the "Share" button at the top right of the notebook.
Share it with us: annabison, giovannidonghi and riccardocappi.
**If you do NOT share your notebook, your submission will be discarded**.