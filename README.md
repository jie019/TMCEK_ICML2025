
# [Trusted Multi-View Classification with Expert Knowledge Constraints](https://openreview.net/pdf?id=U64wEbM7NB)
<div align="center">
  
**[_Xinyan Liang_<sup>1</sup>](https://xinyanliang.github.io/), [_Shijie Wang_<sup>1</sup>](https://jie019.github.io/), [_Yuhua Qian_<sup>1</sup>](http://dig.sxu.edu.cn/qyh/), _Qian Guo_<sup>2</sup>, _Liang Du_<sup>1</sup>, _Bingbing Jiang_<sup>3</sup>, _Tingjin Luo_<sup>4</sup>, _Feijiang Li_<sup>1</sup>**

<sup>1</sup>SXU <sup>2</sup>TYUST <sup>3</sup>HZNU <sup>4</sup>NUDT
</div>
<p align="center">
  <a href="https://openreview.net/pdf?id=U64wEbM7NB">
    <img src="https://img.shields.io/badge/OpenReview-gray" alt="OpenReview">
  </a>
</p>

# Abstract

Trusted multi-view classification (TMVC) based on the Dempster-Shafer theory has gained significant recognition for its reliability in safety-critical applications. However, existing methods predominantly focus on providing confidence levels for decision outcomes without explaining the reasoning behind these decisions.
Moreover, the reliance on first-order statistical magnitudes of belief masses often inadequately capture the intrinsic uncertainty within the evidence. 
To address these limitations, we propose a novel framework termed Trusted Multi-view Classification Constrained with Expert Knowledge (TMCEK). TMCEK integrates expert knowledge to enhance feature-level interpretability and introduces a distribution-aware subjective opinion mechanism to derive more reliable and realistic confidence estimates. The theoretical superiority of the proposed uncertainty measure over conventional approaches is rigorously established. Extensive experiments conducted on three multi-view datasets for sleep stage classification demonstrate that TMCEK achieves state-of-the-art performance while offering interpretability at both the feature and decision levels. These results position TMCEK as a robust and interpretable solution for MVC in safety-critical domains.

# 🏗️Model
<div align="center">
  <img src="model.png" />
</div>

# Experiment 1: Sleep Stage Classification
**Directory Structure**

**Data**

We used three public datasets in this experiment:
- [Sleep-EDF20](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [Sleep-EDF78](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)
  
**Experiment Workflow**

# Experiment 2: Multi-view Classification

**Directory Structure**
bash
Multi-view Classification/
├── data/
│   └── 
├── dataset.py
├── loss_function.py
├── main.py
└── model.py  

**Data**
We used three public datasets in this experiment:
- [HandWritten (HW)](https://archive.ics.uci.edu/dataset/72/multiple+features)
- [Scene15](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177/1)
- [CUB](https://www.vision.caltech.edu/visipedia/CUB-200.html)
- [PIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Home.html)

# 📑Citatio


# 🙏Acknowledgement

# 📬Contact
If you have any detailed questions or suggestions, you can email us: [wshijie0@163.com](mailto:wshijie0@163.com)
