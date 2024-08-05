# VMSVFND
Code for paper["***Viewpoint-sentiment guided multimodal short video fake news detection***"]

### Environment
please refer to the file requirements.txt.

### Data Processing
The training model can directly use the pre-extracted embeddings we extracted.
You could use pre-extracted features ([VGG19](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/ptvgg19_frames.zip)/[C3D](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/c3d.zip)/[VGGish](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/dict_vid_audioconvfea.pkl)).These are all provided by [them](https://github.com/ICTMCG/FakeSV).
You could use our pre-extracted [viewpoint-sentiment-embed](https://drive.google.com/file/d/152CeWGgiH0aviDKTle-DnX0vfA8wU_r-/view?usp=drive_link) and [C3D-SRM](https://drive.google.com/file/d/1--usuXuPoHz3Qnyh1_oNPkyx1811gL5_/view?usp=drive_link) features.
Please place these features in the specified location, which can be customized in dataloader.py.
After placing the data, start training the model:
Pretrained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
```python
python main.py
```
### checkpoint
we provide [checkpoints](https://drive.google.com/file/d/1dE4WG-fShnFMIwAKhGYGbZanlJ9b_svu/view?usp=drive_link) for comparison,and the model trained in the first stage[contrastive_model](https://drive.google.com/file/d/1WQXVMowZDmOE-s6mTz05ZCXEfRTGdZwU/view?usp=drive_link).
Set the path in the test.py and use it:
```python
python test.py
```
### Dataset
The original dataset can be applied for [here](https://github.com/ICTMCG/FakeSV) 

