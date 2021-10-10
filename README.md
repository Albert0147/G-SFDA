# [Generalized Source-free Domain Adaptation (ICCV 2021)](https://arxiv.org/abs/2108.01614)
Code (based on **pytorch 1.3, cuda 10.0, please check the 'requirements.txt' for reproducing the results**) for our ICCV 2021 paper 'Generalized Source-free Domain Adaptation'. [[project]](https://sites.google.com/view/g-sfda/g-sfda) [[paper]](https://arxiv.org/abs/2108.01614). 

(Please also check our **NeurIPS** 2021 paper 'Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation'. [[project]](https://sites.google.com/view/trustyourgoodfriend-neurips21/) [[paper (soon)]](), which goes deeper into the neighborhood clustering for SFDA.)


## Dataset preparing
Download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) (use our provided image list files) dataset. And denote the path of data list in the code.


## Training
First train the model on source data with both source and target attention, then adapt the model to target domain in absence of source data. We use embedding layer to automatically produce the domain attention.
> sh visda.sh (for VisDA)\
> sh office-home.sh (for Office-Home)

**Checkpoints** We provide the training log files, source model and target model on VisDA in this [link](https://drive.google.com/drive/folders/1QrK_oDWbSAXdLzICUhSc2sdrxlUcXF5n?usp=sharing). You can directly start the source-free adaptation from our source model to reproduce the results.

## Domain Classifier
The file 'domain_classifier.ipynb' contains the code for training domain classifier and evaluating the model with estimated domain ID (on VisDA).

### Acknowledgement

The codes are based on [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT).
