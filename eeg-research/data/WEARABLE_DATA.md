# Dataset availability for seizure detection and forecasting from non-EEG signals, using wearable devices

*Author: Jozsef Kovacs*  
*Created: Nov 13th, 2023*  
*Modified: Feb 12th, 2024*

The purpose of this document is to provide an overview of non-EEG datasets obtained from wearable devices and used in 
research projects of epileptic seizure prediction or forecasting, which are or should become accessible for experimental 
purposes. The datasets come from a wide variety of data-acquisiton settings with reagards to the devices involved, the 
monitoring environment of the patients, trial setup and the source of confirmation/control (self-reported, video-EEG, 
iEEG, ECoG/ECG, in-patient observation or some combination). This makes the different datasets difficult to compare 
directly.

After the overview, there is a list of contacts and useful links, which allow for accessing or submitting a request to access the different datasets.

## Overview of projects and datasets

In late 2023, the following research projects related to epileptic seizure detection and forecasting 
via non-EEG wearables had already disclosed datasets, or had announced their intention to make 
their experimental datasets available (publicly or upon request) in the foreseeable future:

| Project                                                   | Institution / Source                                                                              | Availability<br/>status     | Type/device(s)                                           |                                                                                    Subjects (duration) | Contact / responsible                                             |
|:----------------------------------------------------------|---------------------------------------------------------------------------------------------------|:---------------------------:|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------:|-------------------------------------------------------------------|
| PreEpiSeizures<br/>(2023)                                 | Instituto de Telecomunicações<br/>Instituto Superior Técnico, Universidade de Lisboa              | expected 2024               | ChestBIT<br/>WristBIT<br/>ArmBIT<br/>(labels: video-EEG) | 59 patients<br/>(ChestBIT: 37)<br/>nearly 6,000 hours                                                  | Mariana Abreu<br/>University of Lisbon<br/>Portugal               |
| Wristbands for reliable real-time predictions (2023-2024) | Paderborn University<br/>Germany                                                                  | expected 2024               | n/a (?Empatica Embrace)                                  | n/a                                                                                                    | Tanuj Hasija<br/>Paderborn University<br/>Germany                 |
| My Seizure Gauge<br/>(2022-2023)                          | Epilepsy Foundation<br/>(Epilepsy Ecosystem)                                                      | upon registration & request | Empatica E4<br/>Byteflies<br/>Epilog<br/>(labels: iEEG)  | 27 patients<br/>(1-9 days)                                                                             | Levin Kuhlmann<br/>Monash University<br/>Australia                |
| Wearable Seizure Forecasting Pilot<br/>(2023)             | University of Melbourne                                                                           | public                      | Fitbit<br/>Seer App                                      | 13 patients<br/>(562 days smartwatch<br/>125 days app,<br/>mean duration)                              | Pip Karoly<br/>University of Melbourne<br/>Australia              |
| Seizure Diary + Heart rate<br/>(2021)                     | University of Melbourne                                                                           | public                      | Fitbit<br/>Seer App                                      | 46 patients<br/>(31 with epilepsy<br/>12.0+/-5.9 months,<br/>15 healthy/control<br/>10.6+/-6.4 months) | Pip Karoly<br/>University of Melbourne<br/>Australia              |
| DL from wristband sensor data<br/>(2019)                  | Boston Children’s Hospital, Boston, USA<br/>University Clinic Carl Gustav Carus, Dresden, Germany | upon request                | Empatica E4<br/>(labels: video-EEG)                      | 50 patients<br/> 1,400 hours                             | Christian Meisel<br/>University Clinic Carl Gustav Carus, Dresden |

---

## Useful links and further information on obtaining the datasets

### My Seizure Gauge (2022-2023)
*[Epilepsy Foundation / epilepsyecosystem.org]*

#### Obtaining the dataset:  
Instructions are available on [this webpage](https://www.epilepsyecosystem.org/my-seizure-gauge-1#data-2). A registration with epilepsyecosystem.org is necessary, as part of 
the confirmation process an email with detailed instructions will be sent to you. Upon completion of the registration process, as soon as they activate your account,
you will be able to access the dataset via the Australian Seer Medical portal.  

For downloading the available MSG dataset, it is recommended to clone the [seer-py github project](https://github.com/seermedical/seer-py) and use the downloader script
in the `seer-py/Examples` folder called `msg_data_downloader.py`. This will download the '.parquet' files, labels, as well as the metadata describing the file content.
After starting the initial download, your account ID and passwords will be requested. Also a studies.json config file is created with the list of avialable studies, and
the status of their download (0: unprocessed, 1: downloaded). If necessary, you can edit this file to change the set of subjects/studies being downloaded in one session. 
Downloading can take a while, and will require approximately 70GB of space (if only the available MSG trial data is downloaded).

#### Contact information:  
**Levin Kuhlmann, PhD**  
Associate Professor of Data Science, AI and Digital Health  
Director of Master of AI  
Faculty of Information Technology  
Monash University  
Rm 2.73, Woodside Building, Clayton Campus  
email: levin.kuhlmann@monash.edu  
tel: +61412 552283  
web: https://research.monash.edu/en/persons/levin-kuhlmann    

#### Related paper(s):

* [Nasseri, M., Pal Attia, T., Joseph, B., Gregg, N. M., Nurse, E. S., Viana, P. F., ... & Brinkmann, B. H. (2021). Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning. Scientific reports, 11(1), 21935.](https://www.nature.com/articles/s41598-021-01449-2)  
* [Brinkmann, B., Nurse, E., Viana, P., Nasseri, M., Kuhlmann, L., Karoly, P., ... & Freestone, D. (2023). Seizure Forecasting and Detection with Wearable Devices and Subcutaneous EEG–Outcomes from the My Seizure Gauge Trial (PL4. 001).](https://www.neurology.org/doi/abs/10.1212/WNL.0000000000203901)
* [Schlegel, K., Kleyko, D., Brinkmann, B.H. et al. (2024). Lessons from a challenge on forecasting epileptic seizures from non-cerebral signals. Nat Mach Intell 6, 243–244 (2024).](https://doi.org/10.1038/s42256-024-00799-6)

#### Other resources:
* [eval.ai - My Seizure Gauge challenge - *NB. not the same dataset structure/content as from Seer Medical*](https://eval.ai/web/challenges/challenge-page/1693/overview)  
* [Ei2: Seizure Gauge Challenge](https://www.epilepsy.com/research-funding/epilepsy-innovation-institute/seizure-gauge-challenge)  
* [Seer Research](https://seermedical.com/au/research-and-publications/lessons-from-a-challenge-on-forecasting-epileptic-seizures-from-non-cerebral-signals)

---

### Wearable Seizure Forecasting (2023) & Seizure Diary / Heart rate (2021) 
*[Seer Medical / The University of Melbourne]*

#### Obtaining the dataset:  
Datasets related to these studies were shared by Dr. Philippa Karoly and can be accessed following [this link](https://melbourne.figshare.com/authors/PHILIPPA_KAROLY/1239060).  

For the Wearable Seizure Forecasting (2023) study with 13 patients the heart rate data and the labels are in `csv` 
format. The heart rate files (heart_rate_XX.csv) contain a timestamp and a beet-per-minute value,
while the label (id_XX.csv) files correspond to the timestamps of the reported events.  

For the Seizure Diary, Heart rage (2021) study the heart rate data and labels are as well in `csv` format, and the 
structure corresponds to one described above, with the exception that the timestamps are in relative 
time measured from the start of the recording.

#### Contact information:  
**Dr. Pip Karoly**  
senior lecturer, Biomedical Engineering  
The University of Melbourne  
email: [karoly.p@unimelb.edu.au](mailto:karoly.p@unimelb.edu.au)

#### Related paper(s):  
* [Stirling, R. E., Grayden, D. B., D'Souza, W., Cook, M. J., Nurse, E., Freestone, D. R., ... & Karoly, P. J. (2021). Forecasting seizure likelihood with wearable technology. Frontiers in neurology, 12, 704060.](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2021.704060/full)  
* [Xiong, W., Stirling, R. E., Payne, D. E., Nurse, E. S., Kameneva, T., Cook, M. J., ... & Karoly, P. J. (2023). Forecasting seizure likelihood from cycles of self-reported events and heart rate: a prospective pilot study. EBioMedicine, 93.](https://doi.org/10.1016/j.ebiom.2023.104656)

#### Other links and resources:  
* [github - ictals primer notebook](https://github.com/pkaroly/ictals_primer_notebook)  
* [video - Dr Pip Karoly interview for National Science Week](https://www.youtube.com/watch?v=VzjrfOYN5VQ)
* [video - Seer Medical / Pip Karoly - The link between heart rate cycles and seizure timing](https://www.youtube.com/watch?v=VcsM8-3Fqmk)

---

### PreEpiSeizures (2023)

*NB: This dataset is not yet available, but is expected to become accessible in 2024, when the legal background will be
finalised. Administrative/technical conditions of downloading the dataset have not yet been disclosed.*

#### Contact information:  
**Mariana Abreu**  
IT - Lisboa - Instituto Superior Técnico  
email: [mariana.abreu@lx.it.pt](mailto:mariana.abreu@lx.it.pt)  
email: [mariana.abreu@tecnico.ulisboa.pt](mailto:mariana.abreu@tecnico.ulisboa.pt)  
tel: +351 21 841 84 54  
fax: +351 21 841 84 72  
github: MarianaAbreu

#### Related paper(s):
* [Abreu et al. (2023) - PreEpiSeizures: description and outcomes of physiological data acquisition using wearable devices during video-EEG monitoring in people with epilepsy](https://www.frontiersin.org/articles/10.3389/fneur.2021.740743)

#### Other links and resources:  
* [*video:* PreEpiSeizures: novel tools to diagnose and monitor epilepsy](https://www.youtube.com/watch?v=AVjnpqmTkIg)
* [preepiseizures (github project)](https://github.com/MarianaAbreu/preepiseizures)

--- 

### Wristbands for reliable real-time predictions (2023-2024)
*[University of Paderborn, Germany]*

*NB: Project started in October 2023, this the dataset is not yet available.*

#### Contact information:  
**Dr. Tanuj Hasija**  
Signal and System Theory Group  
Dept. of Electrical Engineering and Information Technology (EIM-E)  
Universität Paderborn  
email: [tanuj.hasija@sst.upb.de](mailto:tanuj.hasija@sst.upb.de)  
tel: +49 5251 60-3181  
fax: +49 5251 60-2989  
web: https://sst-group.org/team/tanuj-hasija/  

#### Related paper(s):
* [Vieluf, S., Hasija, T., Kuschel, M., Reinsberger, C., & Loddenkemper, T. (2023). Developing a deep canonical correlation-based technique for seizure prediction. Expert Systems with Applications, 234, 120986.](https://www.sciencedirect.com/science/article/pii/S0957417423014884)  
* [Vieluf, S., Hasija, T., Schreier, P. J., El Atrache, R., Hammond, S., Touserkani, F. M., ... & Reinsberger, C. (2021). Generalized tonic-clonic seizures are accompanied by changes of interrelations within the autonomic nervous system. Epilepsy & Behavior, 124, 108321.](https://pubmed.ncbi.nlm.nih.gov/34624803/)  

#### Other resources:  
[*university - press release 11/10/2023:* Preventing epileptic seizures: Wristbands for reliable real-time predictions](https://www.uni-paderborn.de/en/news-item/127266)  

--- 

### DL from wristband sensor data (2019)
*[Boston Children’s Hospital, Boston, USA / University Clinic Carl Gustav Carus, Dresden, Germany]*

*NB: Prof.Dr. Christian Meisel is provided as contact in the related research papers, but regarding the 2019 study the
data from the research belongs to Prof. Tobias Loddenkemper, who should be approached and contacted with such requests.*

#### Contact information:
* research paper:  
**PD Dr. med. Christian Meisel**  
email: [crishen@yahoo.com](mailto:crishen@yahoo.com)


* dataset:  
**Prof. Tobias Loddenkemper, MD**  
Professor of Neurology  
Director, Clinical Epilepsy Research  
Harvard Medical School   
email: [tobias.loddenkemper@childrens.harvard.edu](mailto:tobias.loddenkemper@childrens.harvard.edu)  

#### Related paper(s):
* [Meisel, C., Atrache, R. E., Jackson, M., Schubach, S., Ufongene, C., & Loddenkemper, T. (2019). Deep learning from wristband sensor data: towards wearable, non-invasive seizure forecasting. arXiv preprint arXiv:1906.00511.](https://arxiv.org/abs/1906.00511)  

#### Other resources:  
n/a

---

## Information on devices used for data acquisition  

Wearable devices used for data acquisition in the above-listed studies:  
* [Empatica E4](https://empatica.app.box.com/v/E4-User-Manual) 
* [Byteflies Kit - Model 1](https://ifu.byteflies.net/docs/hcp/kit.html)
* [Epilog / Epitel](https://www.epitel.com/)
* [BITaliano - ArmBIT, WristBIT, ChestBIT](https://www.pluxbiosignals.com/collections/bitalino)

EEG device for confirmation/control:  
* [NeuroPace RNS (R)](https://www.accessdata.fda.gov/cdrh_docs/pdf10/p100026c.pdf)













