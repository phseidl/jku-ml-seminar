# Some links for papers
-- add your name if you pick it ;) e.g. 

- Markus Gutenberger [CLEEGN: A Convolutional Neural Network for Plug-and-Play Automatic EEG Reconstruction](https://openreview.net/forum?id=O3NQgGoGu6)
- Liza Shchasnovich [BIOT: Cross-data Biosignal Learning in the Wild](https://arxiv.org/abs/2305.10351)
- Jozsef Kovacs [Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning](https://www.nature.com/articles/s41598-021-01449-2)
- Ahmed Mohammed [ThoughtViz: Visualizing Human Thoughts Using Generative
Adversarial Network](https://dl.acm.org/doi/abs/10.1145/3240508.3240641)

## Papers

[Submitted on 3 Jan 2023]
Unsupervised Multivariate Time-Series Transformers for Seizure Identification on EEG
İlkay Yıldız Potter, George Zerveas, Carsten Eickhoff, Dominique Duncan
Epilepsy is one of the most common neurological disorders, typically observed via seizure episodes. Epileptic seizures are commonly monitored through electroencephalogram (EEG) recordings due to their routine and low expense collection. The stochastic nature of EEG makes seizure identification via manual inspections performed by highly-trained experts a tedious endeavor, motivating the use of automated identification. The literature on automated identification focuses mostly on supervised learning methods requiring expert labels of EEG segments that contain seizures, which are difficult to obtain. Motivated by these observations, we pose seizure identification as an unsupervised anomaly detection problem. To this end, we employ the first unsupervised transformer-based model for seizure identification on raw EEG. We train an autoencoder involving a transformer encoder via an unsupervised loss function, incorporating a novel masking strategy uniquely designed for multivariate time-series data such as EEG. Training employs EEG recordings that do not contain any seizures, while seizures are identified with respect to reconstruction errors at inference time. We evaluate our method on three publicly available benchmark EEG datasets for distinguishing seizure vs. non-seizure windows. Our method leads to significantly better seizure identification performance than supervised learning counterparts, by up to 16% recall, 9% accuracy, and 9% Area under the Receiver Operating Characteristics Curve (AUC), establishing particular benefits on highly imbalanced data. Through accurate seizure identification, our method could facilitate widely accessible and early detection of epilepsy development, without needing expensive label collection or manual feature extraction.
https://arxiv.org/abs/2301.03470
https://github.com/ilkyyldz95/EEG_MVTS

Neural decoding of music from the EEG
Abstract
Neural decoding models can be used to decode neural representations of visual, acoustic, or semantic information. Recent studies have demonstrated neural decoders that are able to decode accoustic information from a variety of neural signal types including electrocortiography (ECoG) and the electroencephalogram (EEG). In this study we explore how functional magnetic resonance imaging (fMRI) can be combined with EEG to develop an accoustic decoder. Specifically, we first used a joint EEG-fMRI paradigm to record brain activity while participants listened to music. We then used fMRI-informed EEG source localisation and a bi-directional long-term short term deep learning network to first extract neural information from the EEG related to music listening and then to decode and reconstruct the individual pieces of music an individual was listening to. We further validated our decoding model by evaluating its performance on a separate dataset of EEG-only recordings. We were able to reconstruct music, via our fMRI-informed EEG source analysis approach, with a mean rank accuracy of 71.8%. Using only EEG data, without participant specific fMRI-informed source analysis, we were able to identify the music a participant was listening to with a mean rank accuracy of 59.2%. This demonstrates that our decoding model may use fMRI-informed source analysis to aid EEG based decoding and reconstruction of acoustic information from brain activity and makes a step towards building EEG-based neural decoders for other complex information domains such as other acoustic, visual, or semantic information.
https://www.nature.com/articles/s41598-022-27361-x#data-availability

Epilepsy Detection by Using Scalogram Based Convolutional Neural Network from EEG Signals
Ömer Türk1,* and Mehmet Siraç Özerdem2
Author information Article notes Copyright and License information PMC Disclaimer
Abstract
The studies implemented with Electroencephalogram (EEG) signals are progressing very rapidly and brain computer interfaces (BCI) and disease determinations are carried out at certain success rates thanks to new methods developed in this field. The effective use of these signals, especially in disease detection, is very important in terms of both time and cost. Currently, in general, EEG studies are used in addition to conventional methods as well as deep learning networks that have recently achieved great success. The most important reason for this is that in conventional methods, increasing classification accuracy is based on too many human efforts as EEG is being processed, obtaining the features is the most important step. This stage is based on both the time-consuming and the investigation of many feature methods. Therefore, there is a need for methods that do not require human effort in this area and can learn the features themselves. Based on that, two-dimensional (2D) frequency-time scalograms were obtained in this study by applying Continuous Wavelet Transform to EEG records containing five different classes. Convolutional Neural Network structure was used to learn the properties of these scalogram images and the classification performance of the structure was compared with the studies in the literature. In order to compare the performance of the proposed method, the data set of the University of Bonn was used. The data set consists of five EEG records containing healthy and epilepsy disease which are labeled as A, B, C, D, and E. In the study, A-E and B-E data sets were classified as 99.50%, A-D and B-D data sets were classified as 100% in binary classifications, A-D-E data sets were 99.00% in triple classification, A-C-D-E data sets were 90.50%, B-C-D-E data sets were 91.50% in quaternary classification, and A-B-C-D-E data sets were in the fifth class classification with an accuracy of 93.60%.

From <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6562774/> 


Transformers for EEG classification : architectures, pre-training, and applications to epileptic seizure forecasting
Bary_10511800_2023.pd 
By : Bary, Tim [UCL]Directed by : Macq, Benoît [UCL]
Used at first in the fields of Natural Language Processing (NLP) and Computer Vision (CV), the transformer neural network recently became popular for electroencephalogram (EEG) signals analysis. However, despite being appreciated for their ability to understand long-term dependencies and their great training speed derived from their parallelizability, transformers require a large amount of labelled data to train effectively. Such data is often scarce in the medical field, as annotations made by experts are costly. This is why self-supervised training, using unlabelled data, has to be performed beforehand. On top of providing a non-exhaustive list of transformer architectures used for EEG classification, we present a way to design several datasets from unlabeled EEG data, which can then be used to pre-train transformers to learn representations of EEG signals. We tested this method on an epileptic seizure forecasting task on the TUSZ dataset (https://doi.org/10.3389/fninf.2018.00083) using a Multi-channel Vision Transformer (MViT, https://doi.org/10.3390/biomedicines10071551). Our results suggest that models pre-trained with this approach not only train significantly faster, but also yield better performances than models that are not pre-trained. The code produced during this thesis is available on this repository: https://github.com/tbary/MasterThesis.git.

From <https://dial.uclouvain.be/memoire/ucl/en/object/thesis%3A40381> 


EEG datasets for seizure detection and prediction— A review
Sheng Wong,

 1 Anj Simmons, 1 Jessica Rivera?Villicana, 1 Scott Barnett, 1 Shobi Sivathamboo, 2 , 3 , 4 , 5 Piero Perucca, 3 , 4 , 5 , 6 , 7 Zongyuan Ge, 8 Patrick Kwan, 4 , 5 Levin Kuhlmann, 9 , 10 Rajesh Vasa, 1 Kon Mouzakis, 1 and Terence J. O'Brien 2 , 3 , 4 , 5
Electroencephalogram (EEG) datasets from epilepsy patients have been used to develop seizure detection and prediction algorithms using machine learning (ML) techniques with the aim of implementing the learned model in a device. However, the format and structure of publicly available datasets are different from each other, and there is a lack of guidelines on the use of these datasets. This impacts the generatability, generalizability, and reproducibility of the results and findings produced by the studies. In this narrative review, we compiled and compared the different characteristics of the publicly available EEG datasets that are commonly used to develop seizure detection and prediction algorithms. We investigated the advantages and limitations of the characteristics of the EEG datasets. Based on our study, we identified 17 characteristics that make the EEG datasets unique from each other. We also briefly looked into how certain characteristics of the publicly available datasets affect the performance and outcome of a study, as well as the influences it has on the choice of ML techniques and preprocessing steps required to develop seizure detection and prediction algorithms. In conclusion, this study provides a guideline on the choice of publicly available EEG datasets to both clinicians and scientists working to develop a reproducible, generalizable, and effective seizure detection and prediction algorithm.
From <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10235576/> 


Interpretable and Robust AI in EEG Systems: A Survey 
Xinliang Zhou1 , Chenyu Liu1 , Liming Zhai1 , Ziyu Jia2,3 , Cuntai Guan1 , Yang Liu
The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI’s reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.
2304.10755.pdf (arxiv.org)

Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning
Scientific Reports volume 11, Article number: 21935 (2021) Cite this article
• 10k Accesses
• Metricsdetails
Abstractsup
The ability to forecast seizures minutes to hours in advance of an event has been verified using invasive EEG devices, but has not been previously demonstrated using noninvasive wearable devices over long durations in an ambulatory setting. In this study we developed a seizure forecasting system with a long short-term memory (LSTM) recurrent neural network (RNN) algorithm, using a noninvasive wrist-worn research-grade physiological sensor device, and tested the system in patients with epilepsy in the field, with concurrent invasive EEG confirmation of seizures via an implanted recording device. The system achieved forecasting performance significantly better than a random predictor for 5 of 6 patients studied, with mean AUC-ROC of 0.80 (range 0.72–0.92). These results provide the first clear evidence that direct seizure forecasts are possible using wearable devices in the ambulatory setting for many patients with epilepsy.

From <https://www.nature.com/articles/s41598-021-01449-2> 


Weak self-supervised learning for seizure forecasting: a feasibility study
Yikai Yang 1, Nhan Duy Truong 1 2, Jason K Eshraghian 3, Armin Nikpour 4 5, Omid Kavehei 1 2
Affiliations expand
• PMID: 35950196
 
• PMCID: PMC9346358
 
• DOI: 10.1098/rsos.220374
Free PMC article
Abstract
This paper proposes an artificial intelligence system that continuously improves over time at event prediction using initially unlabelled data by using self-supervised learning. Time-series data are inherently autocorrelated. By using a detection model to generate weak labels on the fly, which are concurrently used as targets to train a prediction model on a time-shifted input data stream, this autocorrelation can effectively be harnessed to reduce the burden of manual labelling. This is critical in medical patient monitoring, as it enables the development of personalized forecasting models without demanding the annotation of long sequences of physiological signal recordings. We perform a feasibility study on seizure prediction, which is identified as an ideal test case, as pre-ictal brainwaves are patient-specific, and tailoring models to individual patients is known to improve forecasting performance significantly. Our self-supervised approach is used to train individualized forecasting models for 10 patients, showing an average relative improvement in sensitivity by 14.30% and a reduction in false alarms by 19.61% in early seizure forecasting. This proof-of-concept on the feasibility of using a continuous stream of time-series neurophysiological data paves the way towards a low-power neuromorphic neuromodulation system.

From <https://pubmed.ncbi.nlm.nih.gov/35950196/> 


Self-Supervised Contrastive Learning for Medical Time Series: A Systematic Review Abstract
Medical time series are sequential data collected over time that measures health-related signals, such as electroencephalography (EEG), electrocardiography (ECG), and intensive care unit (ICU) readings. Analyzing medical time series and identifying the latent patterns and trends that lead to uncovering highly valuable insights for enhancing diagnosis, treatment, risk assessment, and disease progression. However, data mining in medical time series is heavily limited by the sample annotation which is time-consuming and labor-intensive, and expert-depending. To mitigate this challenge, the emerging self-supervised contrastive learning, which has shown great success since 2020, is a promising solution. Contrastive learning aims to learn representative embeddings by contrasting positive and negative samples without the requirement for explicit labels. Here, we conducted a systematic review of how contrastive learning alleviates the label scarcity in medical time series based on PRISMA standards. We searched the studies in five scientific databases (IEEE, ACM, Scopus, Google Scholar, and PubMed) and retrieved 1908 papers based on the inclusion criteria. After applying excluding criteria, and screening at title, abstract, and full text levels, we carefully reviewed 43 papers in this area. Specifically, this paper outlines the pipeline of contrastive learning, including pre-training, fine-tuning, and testing. We provide a comprehensive summary of the various augmentations applied to medical time series data, the architectures of pre-training encoders, the types of fine-tuning classifiers and clusters, and the popular contrastive loss functions. Moreover, we present an overview of the different data types used in medical time series, highlight the medical applications of interest, and provide a comprehensive table of 51 public datasets that have been utilized in this field. In addition, this paper will provide a discussion on the promising future scopes such as providing guidance for effective augmentation design, developing a unified framework for analyzing hierarchical time series, and investigating methods for processing multimodal data. Despite being in its early stages, self-supervised contrastive learning has shown great potential in overcoming the need for expert-created annotations in the research of medical time series.

From <https://www.mdpi.com/1424-8220/23/9/4221> 



Time Series as Images: Vision Transformer for Irregularly Sampled Time Series 
Zekun Li, Shiyang Li, Xifeng Yan
Published: 01 Mar 2023, Last Modified: 22 Apr 2023ICLR 2023 TSRL4H OralReaders:  EveryoneShow BibtexShow Revisions
Keywords: irregularly sampled time series, vision transformer, healthcare, time series classification, multivariate time series
TL;DR: We introduce a new perspective for irregularly sampled time series modeling, i.e., transforming time series data into line graph images and utilizing vision transformers to perform downstream tasks, which achieves SOTA results.
Abstract: Irregularly sampled time series are becoming increasingly prevalent in various domains, especially medical applications. Although different highly-customized methods have been proposed to tackle irregularity, how to effectively model their complicated dynamics and high sparsity is still an open problem. This paper studies the problem from a whole new perspective: transforming irregularly sampled time series into line graph images and adapting powerful vision transformers to perform time series classification in the same way as image classification. Our approach largely simplifies algorithm designs without assuming prior knowledge and can be potentially extended as a general-purpose framework. Despite its simplicity, we show that it substantially outperforms state-of-the-art specialized algorithms on several popular healthcare and human activity datasets. Our code and data are available at \url{https://github.com/Leezekun/ViTST}.

From <https://openreview.net/forum?id=Nv0tzncECS> 


[Submitted on 14 Jun 2023]
Towards trustworthy seizure onset detection using workflow notes
Khaled Saab, Siyi Tang, Mohamed Taha, Christopher Lee-Messer, Christopher Ré, Daniel Rubin
	A major barrier to deploying healthcare AI models is their trustworthiness. One form of trustworthiness is a model's robustness across different subgroups: while existing models may exhibit expert-level performance on aggregate metrics, they often rely on non-causal features, leading to errors in hidden subgroups. To take a step closer towards trustworthy seizure onset detection from EEG, we propose to leverage annotations that are produced by healthcare personnel in routine clinical workflows -- which we refer to as workflow notes -- that include multiple event descriptions beyond seizures. Using workflow notes, we first show that by scaling training data to an unprecedented level of 68,920 EEG hours, seizure onset detection performance significantly improves (+12.3 AUROC points) compared to relying on smaller training sets with expensive manual gold-standard labels. Second, we reveal that our binary seizure onset detection model underperforms on clinically relevant subgroups (e.g., up to a margin of 6.5 AUROC points between pediatrics and adults), while having significantly higher false positives on EEG clips showing non-epileptiform abnormalities compared to any EEG clip (+19 FPR points). To improve model robustness to hidden subgroups, we train a multilabel model that classifies 26 attributes other than seizures, such as spikes, slowing, and movement artifacts. We find that our multilabel model significantly improves overall seizure onset detection performance (+5.9 AUROC points) while greatly improving performance among subgroups (up to +8.3 AUROC points), and decreases false positives on non-epileptiform abnormalities by 8 FPR points. Finally, we propose a clinical utility metric based on false positives per 24 EEG hours and find that our multilabel model improves this clinical utility metric by a factor of 2x across different clinical settings.

From <https://arxiv.org/abs/2306.08728> 


[Submitted on 15 Jun 2023]
MBrain: A Multi-channel Self-Supervised Learning Framework for Brain Signals
Donghong Cai, Junru Chen, Yang Yang, Teng Liu, Yafeng Li
	Brain signals are important quantitative data for understanding physiological activities and diseases of human brain. Most existing studies pay attention to supervised learning methods, which, however, require high-cost clinical labels. In addition, the huge difference in the clinical patterns of brain signals measured by invasive (e.g., SEEG) and non-invasive (e.g., EEG) methods leads to the lack of a unified method. To handle the above issues, we propose to study the self-supervised learning (SSL) framework for brain signals that can be applied to pre-train either SEEG or EEG data. Intuitively, brain signals, generated by the firing of neurons, are transmitted among different connecting structures in human brain. Inspired by this, we propose MBrain to learn implicit spatial and temporal correlations between different channels (i.e., contacts of the electrode, corresponding to different brain areas) as the cornerstone for uniformly modeling different types of brain signals. Specifically, we represent the spatial correlation by a graph structure, which is built with proposed multi-channel CPC. We theoretically prove that optimizing the goal of multi-channel CPC can lead to a better predictive representation and apply the instantaneou-time-shift prediction task based on it. Then we capture the temporal correlation by designing the delayed-time-shift prediction task. Finally, replace-discriminative-learning task is proposed to preserve the characteristics of each channel. Extensive experiments of seizure detection on both EEG and SEEG large-scale real-world datasets demonstrate that our model outperforms several state-of-the-art time series SSL and unsupervised models, and has the ability to be deployed to clinical practice.

From <https://arxiv.org/abs/2306.13102> 


[Submitted on 15 Jun 2023]
BrainNet: Epileptic Wave Detection from SEEG with Hierarchical Graph Diffusion Learning
Junru Chen, Yang Yang, Tao Yu, Yingying Fan, Xiaolong Mo, Carl Yang
	Epilepsy is one of the most serious neurological diseases, affecting 1-2% of the world's population. The diagnosis of epilepsy depends heavily on the recognition of epileptic waves, i.e., disordered electrical brainwave activity in the patient's brain. Existing works have begun to employ machine learning models to detect epileptic waves via cortical electroencephalogram (EEG). However, the recently developed stereoelectrocorticography (SEEG) method provides information in stereo that is more precise than conventional EEG, and has been broadly applied in clinical practice. Therefore, we propose the first data-driven study to detect epileptic waves in a real-world SEEG dataset. While offering new opportunities, SEEG also poses several challenges. In clinical practice, epileptic wave activities are considered to propagate between different regions in the brain. These propagation paths, also known as the epileptogenic network, are deemed to be a key factor in the context of epilepsy surgery. However, the question of how to extract an exact epileptogenic network for each patient remains an open problem in the field of neuroscience. To address these challenges, we propose a novel model (BrainNet) that jointly learns the dynamic diffusion graphs and models the brain wave diffusion patterns. In addition, our model effectively aids in resisting label imbalance and severe noise by employing several self-supervised learning tasks and a hierarchical framework. By experimenting with the extensive real SEEG dataset obtained from multiple patients, we find that BrainNet outperforms several latest state-of-the-art baselines derived from time-series analysis.

From <https://arxiv.org/abs/2306.13101> 


Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting 
Yunhao Zhang, Junchi Yan
Published: 01 Feb 2023, Last Modified: 02 Mar 2023ICLR 2023 notable top 5%Readers:  EveryoneShow BibtexShow Revisions
Keywords: Transformer, multivariate time series forecasting, deep learning
TL;DR: We propose Crossformer, a Transformer-based model that explicitly utilizes cross-dimension dependency for multivariate time series forecasting.
Abstract: Recently many deep models have been proposed for multivariate time series (MTS) forecasting. In particular, Transformer-based models have shown great potential because they can capture long-term dependency. However, existing Transformer-based models mainly focus on modeling the temporal dependency (cross-time dependency) yet often omit the dependency among different variables (cross-dimension dependency), which is critical for MTS forecasting. To fill the gap, we propose Crossformer, a Transformer-based model utilizing cross-dimension dependency for MTS forecasting. In Crossformer, the input MTS is embedded into a 2D vector array through the Dimension-Segment-Wise (DSW) embedding to preserve time and dimension information. Then the Two-Stage Attention (TSA) layer is proposed to efficiently capture the cross-time and cross-dimension dependency. Utilizing DSW embedding and TSA layer, Crossformer establishes a Hierarchical Encoder-Decoder (HED) to use the information at different scales for the final forecasting. Extensive experimental results on six real-world datasets show the effectiveness of Crossformer against previous state-of-the-arts.

From <https://openreview.net/forum?id=vSVLM2j9eie> 




CLEEGN: A Convolutional Neural Network for Plug-and-Play Automatic EEG Reconstruction 
Pin-Hua Lai, Wei-Chun Yang, Hsiang-Chieh Tsou, Chun-Shu Wei
Published: 01 Mar 2023, Last Modified: 22 Apr 2023ICLR 2023 TSRL4H PosterReaders:  EveryoneShow BibtexShow Revisions
Keywords: EEG, Artifact Removal, Signal Reconstruction
Abstract: Human electroencephalography (EEG) is a brain monitoring modality that senses cortical neuroelectrophysiological activity in high-temporal resolution. One of the greatest challenges posed in applications of EEG is the unstable signal quality susceptible to inevitable artifacts during recordings. To date, most existing techniques for EEG artifact removal and reconstruction are applicable to offline analysis solely, or require individualized training data to facilitate online reconstruction. We have proposed CLEEGN, a light-weight convolutional neural network for plug-and-play automatic EEG reconstruction. CLEEGN is based on a subject-independent pre-trained model using existing data and can operate on a new user without any further calibration. The results of simulated online validation suggest that, even without any calibration, CLEEGN can largely preserve inherent brain activity and outperforms leading online/offline artifact removal methods in the decoding accuracy of reconstructed EEG data. 

From <https://openreview.net/forum?id=O3NQgGoGu6> 


SPP-EEGNET: An Input-Agnostic Self-supervised EEG Representation Model for Inter-dataset Transfer Learning
Part of the Lecture Notes in Networks and Systems book series (LNNS,volume 453)
Abstract
There is currently a scarcity of labeled Electroencephalography (EEG) recordings, and different datasets usually have incompatible setups (e.g., various sampling rates, number of channels, event lengths, etc.). These issues hinder machine learning practitioners from training general-purpose EEG models that can be reused on specific EEG classification tasks through transfer learning. We present a deep convolutional neural network architecture with a spatial pyramid pooling layer that is able to take in EEG signals of varying dimensionality and extract their features to fixed-size vectors. The model is trained with a contrastive self-supervised learning task on a large unlabelled dataset. We introduce a set of EEG signal augmentation techniques to generate large amounts of sample pairs to train the feature extractor. We then transfer the trained feature extractor to new downstream tasks. The experimental results11 show that the first few convolutional layers of the feature extractor learn the general features of the EEG data, which can significantly improve the classification performance on new datasets (The source code on Github:  https://github.com/imics-lab/eeg-transfer-learning).

From <https://link.springer.com/chapter/10.1007/978-3-030-99948-3_17> 


[Submitted on 21 Jan 2022 (v1), last revised 21 Mar 2022 (this version, v2)]
Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting
Kwanhyung Lee, Hyewon Jeong, Seyun Kim, Donghwa Yang, Hoon-Chul Kang, Edward Choi
Electroencephalogram (EEG) is an important diagnostic test that physicians use to record brain activity and detect seizures by monitoring the signals. There have been several attempts to detect seizures and abnormalities in EEG signals with modern deep learning models to reduce the clinical burden. However, they cannot be fairly compared against each other as they were tested in distinct experimental settings. Also, some of them are not trained in real-time seizure detection tasks, making it hard for on-device applications. Therefore in this work, for the first time, we extensively compare multiple state-of-the-art models and signal feature extractors in a real-time seizure detection framework suitable for real-world application, using various evaluation metrics including a new one we propose to evaluate more practical aspects of seizure detection models. Our code is available at this https URL.

From <https://arxiv.org/abs/2201.08780v2> 


[Submitted on 20 Jun 2020 (v1), last revised 22 Oct 2020 (this version, v3)]
wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

From <https://arxiv.org/abs/2006.11477> 


[Submitted on 28 Jan 2021]
BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data
Demetres Kostas, Stephane Aroca-Ouellette, Frank Rudzicz
	Deep neural networks (DNNs) used for brain-computer-interface (BCI) classification are commonly expected to learn general features when trained across a variety of contexts, such that these features could be fine-tuned to specific contexts. While some success is found in such an approach, we suggest that this interpretation is limited and an alternative would better leverage the newly (publicly) available massive EEG datasets. We consider how to adapt techniques and architectures used for language modelling (LM), that appear capable of ingesting awesome amounts of data, towards the development of encephalography modelling (EM) with DNNs in the same vein. We specifically adapt an approach effectively used for automatic speech recognition, which similarly (to LMs) uses a self-supervised training objective to learn compressed representations of raw data signals. After adaptation to EEG, we find that a single pre-trained model is capable of modelling completely novel raw EEG sequences recorded with differing hardware, and different subjects performing different tasks. Furthermore, both the internal representations of this model and the entire architecture can be fine-tuned to a variety of downstream BCI and EEG classification tasks, outperforming prior work in more task-specific (sleep stage classification) self-supervision.

From <https://arxiv.org/abs/2101.12037> 


[Submitted on 24 Apr 2023]
Supervised and Unsupervised Deep Learning Approaches for EEG Seizure Prediction
Zakary Georgis-Yap, Milos R. Popovic, Shehroz S. Khan
	Epilepsy affects more than 50 million people worldwide, making it one of the world's most prevalent neurological diseases. The main symptom of epilepsy is seizures, which occur abruptly and can cause serious injury or death. The ability to predict the occurrence of an epileptic seizure could alleviate many risks and stresses people with epilepsy face. Most of the previous work is focused at seizure detection, we pivot our focus to seizure prediction problem. We formulate the problem of detecting preictal (or pre-seizure) with reference to normal EEG as a precursor to incoming seizure. To this end, we developed several supervised deep learning approaches model to identify preictal EEG from normal EEG. We further develop novel unsupervised deep learning approaches to train the models on only normal EEG, and detecting pre-seizure EEG as an anomalous event. These deep learning models were trained and evaluated on two large EEG seizure datasets in a person-specific manner. We found that both supervised and unsupervised approaches are feasible; however, their performance varies depending on the patient, approach and architecture. This new line of research has the potential to develop therapeutic interventions and save human lives.

From <https://arxiv.org/abs/2304.14922> 


BIOT: Cross-data Biosignal Learning in the Wild
Chaoqi Yang, M. Brandon Westover, Jimeng Sun
	Biological signals, such as electroencephalograms (EEG), play a crucial role in numerous clinical applications, exhibiting diverse data formats and quality profiles. Current deep learning models for biosignals are typically specialized for specific datasets and clinical settings, limiting their broader applicability. Motivated by the success of large language models in text processing, we explore the development of foundational models that are trained from multiple data sources and can be fine-tuned on different downstream biosignal tasks.
	To overcome the unique challenges associated with biosignals of various formats, such as mismatched channels, variable sample lengths, and prevalent missing values, we propose a Biosignal Transformer (\method). The proposed \method model can enable cross-data learning with mismatched channels, variable lengths, and missing values by tokenizing diverse biosignals into unified "biosignal sentences". Specifically, we tokenize each channel into fixed-length segments containing local signal features, flattening them to form consistent "sentences". Channel embeddings and {\em relative} position embeddings are added to preserve spatio-temporal features.
	The \method model is versatile and applicable to various biosignal learning settings across different datasets, including joint pre-training for larger models. Comprehensive evaluations on EEG, electrocardiogram (ECG), and human activity sensory signals demonstrate that \method outperforms robust baselines in common settings and facilitates learning across multiple datasets with different formats. Use CHB-MIT seizure detection task as an example, our vanilla \method model shows 3\% improvement over baselines in balanced accuracy, and the pre-trained \method models (optimized from other data sources) can further bring up to 4\% improvements.

From <https://arxiv.org/abs/2305.10351> 

[Submitted on 16 Jul 2022 (v1), last revised 5 Aug 2022 (this version, v2)]
EEG2Vec: Learning Affective EEG Representations via Variational Autoencoders
David Bethge, Philipp Hallgarten, Tobias Grosse-Puppendahl, Mohamed Kari, Lewis L. Chuang, Ozan Özdenizci, Albrecht Schmidt

There is a growing need for sparse representational formats of human affective states that can be utilized in scenarios with limited computational memory resources. We explore whether representing neural data, in response to emotional stimuli, in a latent vector space can serve to both predict emotional states as well as generate synthetic EEG data that are participant- and/or emotion-specific. We propose a conditional variational autoencoder based framework, EEG2Vec, to learn generative-discriminative representations from EEG data. Experimental results on affective EEG recording datasets demonstrate that our model is suitable for unsupervised EEG modeling, classification of three distinct emotion categories (positive, neutral, negative) based on the latent representation achieves a robust performance of 68.49%, and generated synthetic EEG sequences resemble real EEG data inputs to particularly reconstruct low-frequency signal components. Our work advances areas where affective EEG representations can be useful in e.g., generating artificial (labeled) training data or alleviating manual feature extraction, and provide efficiency for memory constrained edge computing applications.

From <https://arxiv.org/abs/2207.08002> 


[Submitted on 17 May 2023]
EENED: End-to-End Neural Epilepsy Detection based on Convolutional Transformer
Chenyu Liu, Xinliang Zhou, Yang Liu
	Recently Transformer and Convolution neural network (CNN) based models have shown promising results in EEG signal processing. Transformer models can capture the global dependencies in EEG signals through a self-attention mechanism, while CNN models can capture local features such as sawtooth waves. In this work, we propose an end-to-end neural epilepsy detection model, EENED, that combines CNN and Transformer. Specifically, by introducing the convolution module into the Transformer encoder, EENED can learn the time-dependent relationship of the patient's EEG signal features and notice local EEG abnormal mutations closely related to epilepsy, such as the appearance of spikes and the sprinkling of sharp and slow waves. Our proposed framework combines the ability of Transformer and CNN to capture different scale features of EEG signals and holds promise for improving the accuracy and reliability of epilepsy detection. Our source code will be released soon on GitHub.

From <https://arxiv.org/abs/2305.10502> 


[Submitted on 23 May 2023]
Eeg2vec: Self-Supervised Electroencephalographic Representation Learning
Qiushi Zhu, Xiaoying Zhao, Jie Zhang, Yu Gu, Chao Weng, Yuchen Hu
	Recently, many efforts have been made to explore how the brain processes speech using electroencephalographic (EEG) signals, where deep learning-based approaches were shown to be applicable in this field. In order to decode speech signals from EEG signals, linear networks, convolutional neural networks (CNN) and long short-term memory networks are often used in a supervised manner. Recording EEG-speech labeled data is rather time-consuming and laborious, while unlabeled EEG data is abundantly available. Whether self-supervised methods are helpful to learn EEG representation to boost the performance of EEG auditory-related tasks has not been well explored. In this work, we first propose a self-supervised model based on contrastive loss and reconstruction loss to learn EEG representations, and then use the obtained pre-trained model as a feature extractor for downstream tasks. Second, for two considered downstream tasks, we use CNNs and Transformer networks to learn local features and global features, respectively. Finally, the EEG data from other channels are mixed into the chosen EEG data for augmentation. The effectiveness of our method is verified on the EEG match-mismatch and EEG regression tasks of the ICASSP2023 Auditory EEG Challenge.

From <https://arxiv.org/abs/2305.13957> 


[Submitted on 27 Oct 2022]
MAEEG: Masked Auto-encoder for EEG Representation Learning
Hsiang-Yun Sherry Chien, Hanlin Goh, Christopher M. Sandino, Joseph Y. Cheng
	Decoding information from bio-signals such as EEG, using machine learning has been a challenge due to the small data-sets and difficulty to obtain labels. We propose a reconstruction-based self-supervised learning model, the masked auto-encoder for EEG (MAEEG), for learning EEG representations by learning to reconstruct the masked EEG features using a transformer architecture. We found that MAEEG can learn representations that significantly improve sleep stage classification (~5% accuracy increase) when only a small number of labels are given. We also found that input sample lengths and different ways of masking during reconstruction-based SSL pretraining have a huge effect on downstream model performance. Specifically, learning to reconstruct a larger proportion and more concentrated masked signal results in better performance on sleep classification. Our findings provide insight into how reconstruction-based SSL could help representation learning for EEG.

From <https://arxiv.org/abs/2211.02625> 


Abstract
Detecting brain disorders using deep learning methods has received much hype during the last few years. Increased depth leads to more computational efficiency, accuracy, and optimization and less loss. Epilepsy is one of the most common chronic neurological disorders characterized by repeated seizures. We have developed a deep learning model using Deep convolutional Autoencoder—Bidirectional Long Short Memory for Epileptic Seizure Detection (DCAE-ESD-Bi-LSTM) for automatic detection of seizures using EEG data. The significant feature of our model is that it has contributed to the accurate and optimized diagnosis of epilepsy in ideal and real-life situations. The results on the benchmark (CHB-MIT) dataset and the dataset collected by the authors show the relevance of the proposed approach over the baseline deep learning techniques by achieving an accuracy of 99.8%, classification accuracy of 99.7%, sensitivity of 99.8%, specificity and precision of 99.9% and F1 score of 99.6%. Our approach can contribute to the accurate and optimized detection of seizures while scaling the design rules and increasing performance without changing the network’s depth.
Keywords: 
epilepsy; seizure; deep learning; diagnosis; electroencephalogram; Bi-LSTM

From <https://www.mdpi.com/2075-4418/13/4/773> 


Abstract:
Reconstructing images using brain signals of imagined visuals may provide an augmented vision to the disabled, leading to the advancement of Brain-Computer Interface (BCI) technology. The recent progress in deep learning has boosted the study area of synthesizing images from brain signals using Generative Adversarial Networks (GAN). In this work, we have proposed a framework for synthesizing the images from the brain activity recorded by an electroencephalogram (EEG) using small-size EEG datasets. This brain activity is recorded from the subject’s head scalp using EEG when they ask to visualize certain classes of Objects and English characters. We use a contrastive learning method in the proposed framework to extract features from EEG signals and synthesize the images from extracted features using conditional GAN. We modify the loss function to train the GAN, which enables it to synthesize 128 × 128 images using a small number of images. Further, we conduct ablation studies and experiments to show the effectiveness of our proposed framework over other state-of-the-art methods using the small EEG dataset.

From <https://ieeexplore.ieee.org/abstract/document/10096587?casa_token=U5wmKskCB50AAAAA:-UfjCyuPQYmlc5bxeaDU7wtYQjb0pnXdJQUlStJk5z6aq2s8Knk9ChJsTFqWNfGwuHIcpGnhXOCRLA> 

