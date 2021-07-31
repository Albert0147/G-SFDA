# G-SFDA
Code (based on **pytorch 1.3**) for our ICCV 2021 paper ['Generalized Source-free Domain Adaptation'](https://sites.google.com/view/g-sfda/g-sfda).

## Dataset preparing
Download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) dataset. And denote the path of data list in the code.


## Training
First train the model on source data with both source and target attention, then adapt the model to target domain in absence of source data. We use embedding layer to automatically produce the domain attention.
> sh visda.sh (for VisDA)\
> sh office-home.sh (for Office-Home)


## Domain Classifier
The file 'domain_classifier.ipynb' contains the code for training domain classifier and evaluating the model with estimated domain ID (on VisDA).
