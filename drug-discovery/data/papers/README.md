# Some links for papers
-- add your name if you pick it ;) e.g. 

- Philipp Seidl [Time Series as Images: Vision Transformer for Irregularly Sampled Time Series](https://openreview.net/forum?id=Nv0tzncECS)
- Stefan Hangler [COATI: multi-modal contrastive pre-training for representing and traversing chemical space](https://chemrxiv.org/engage/chemrxiv/article-details/64e8137fdd1a73847f73f7aa)

## Papers
[Submitted on 28 Sep 2023]
Language models in molecular discovery
Nikita Janakarajan, Tim Erdmann, Sarath Swaminathan, Teodoro Laino, Jannis Born
	The success of language models, especially transformer-based architectures, has trickled into other domains giving rise to "scientific language models" that operate on small molecules, proteins or polymers. In chemistry, language models contribute to accelerating the molecule discovery cycle as evidenced by promising recent findings in early-stage drug discovery. Here, we review the role of language models in molecular discovery, underlining their strength in de novo drug design, property prediction and reaction chemistry. We highlight valuable open-source software assets thus lowering the entry barrier to the field of scientific language modeling. Last, we sketch a vision for future molecular design that combines a chatbot interface with access to computational chemistry tools. Our contribution serves as a valuable resource for researchers, chemists, and AI enthusiasts interested in understanding how language models can and will be used to accelerate chemical discovery.

From <https://arxiv.org/abs/2309.16235> 


[Submitted on 15 Sep 2023]
Mining Patents with Large Language Models Demonstrates Congruence of Functional Labels and Chemical Structures
Clayton W. Kosonocky, Claus O. Wilke, Edward M. Marcotte, Andrew D. Ellington
	Predicting chemical function from structure is a major goal of the chemical sciences, from the discovery and repurposing of novel drugs to the creation of new materials. Recently, new machine learning algorithms are opening up the possibility of general predictive models spanning many different chemical functions. Here, we consider the challenge of applying large language models to chemical patents in order to consolidate and leverage the information about chemical functionality captured by these resources. Chemical patents contain vast knowledge on chemical function, but their usefulness as a dataset has historically been neglected due to the impracticality of extracting high-quality functional labels. Using a scalable ChatGPT-assisted patent summarization and word-embedding label cleaning pipeline, we derive a Chemical Function (CheF) dataset, containing 100K molecules and their patent-derived functional labels. The functional labels were validated to be of high quality, allowing us to detect a strong relationship between functional label and chemical structural spaces. Further, we find that the co-occurrence graph of the functional labels contains a robust semantic structure, which allowed us in turn to examine functional relatedness among the compounds. We then trained a model on the CheF dataset, allowing us to assign new functional labels to compounds. Using this model, we were able to retrodict approved Hepatitis C antivirals, uncover an antiviral mechanism undisclosed in the patent, and identify plausible serotonin-related drugs. The CheF dataset and associated model offers a promising new approach to predict chemical functionality.

From <https://arxiv.org/abs/2309.08765v1> 


COATI: multi-modal contrastive pre-training for representing and traversing chemical space
25 August 2023, Version 1
Benjamin Kaufman ,
• Creating a successful small molecule drug is a challenging multi-parameter optimization problem in an effectively infinite space of possible molecules. Generative models have emerged as powerful tools for traversing data manifolds comprised of images, sounds, and text, and offer an opportunity to dramatically improve the drug discovery and design process. To create generative optimization methods that are more useful than brute-force molecular generation and filtering via virtual screening, we propose that four integrated features are necessary: large, quantitative datasets of molecular structure and activity, an invertible vector representation of realistic accessible molecules, smooth and differentiable regressors that quantify uncertainty, and algorithms to simultaneously optimize properties of interest. Over the course of 12 months, Terray has collected a dataset of 2 billion quantitative binding measurements, which directly motivates multi-parameter generative optimization of molecules conditioned on this data. To this end, we present COATI, a pre-trained, multi-modal encoder-decoder model of druglike chemical space. COATI is constructed without any human biasing of features, using contrastive learning from text and 3D representations of molecules to allow downstream use with structural models. We demonstrate that COATI possesses many of the desired properties of a universal molecular embedding: fixed-dimension, invertibility, autoencoding, accurate regression, and low computation cost. Finally, we present a novel metadynamics algorithm for generative optimization using a small subset of our proprietary data collected for a model protein, Carbonic Anhydrase, designing molecules that satisfy the multi-parameter optimization task of potency, solubility, and druglikeness. This work sets the stage for fully-integrated generative molecular design and optimization for small molecules.
CiteThis Article

From <https://chemrxiv.org/engage/chemrxiv/article-details/64e8137fdd1a73847f73f7aa> 



MolFM: A Multimodal Molecular Foundation Model 
Yizhen Luo1 , Kai Yang1 , Massimo Hong1,2 , Xingyi Liu1 , Zaiqing Nie1,
Molecular knowledge resides within three different modalities of information sources: molecular structures, biomedical documents, and knowledge bases. Effective incorporation of molecular knowledge from these modalities holds paramount significance in facilitating biomedical research. However, existing multimodal molecular foundation models exhibit limitations in capturing intricate connections between molecular structures and texts, and more importantly, none of them attempt to leverage a wealth of molecular expertise derived from knowledge graphs. In this study, we introduce MolFM, a multimodal molecular foundation model designed to facilitate joint representation learning from molecular structures, biomedical texts, and knowledge graphs. We propose cross-modal attention between atoms of molecular structures, neighbors of molecule entities and semantically related texts to facilitate cross-modal comprehension. We provide theoretical analysis that our cross-modal pre-training captures local and global molecular knowledge by minimizing the distance in the feature space between different modalities of the same molecule, as well as molecules sharing similar structures or functions. MolFM achieves state-of-the-art performance on various downstream tasks. On cross-modal retrieval, MolFM outperforms existing models with 12.13% and 5.04% absolute gains under the zero-shot and fine-tuning settings, respectively. Furthermore, qualitative analysis showcases MolFM’s implicit ability to provide grounding from molecular substructures and knowledge graphs. Code and models are available on https://github.com/BioFM/OpenBioMed

[Submitted on 12 Sep 2022]
A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language
Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen
	Although artificial intelligence (AI) has made significant progress in understanding molecules in a wide range of fields, existing models generally acquire the single cognitive ability from the single molecular modality. Since the hierarchy of molecular knowledge is profound, even humans learn from different modalities including both intuitive diagrams and professional texts to assist their understanding. Inspired by this, we propose a molecular multimodal foundation model which is pretrained from molecular graphs and their semantically related textual data (crawled from published Scientific Citation Index papers) via contrastive learning. This AI model represents a critical attempt that directly bridges molecular graphs and natural language. Importantly, through capturing the specific and complementary information of the two modalities, our proposed model can better grasp molecular expertise. Experimental results show that our model not only exhibits promising performance in cross-modal tasks such as cross-modal retrieval and molecule caption, but also enhances molecular property prediction and possesses capability to generate meaningful molecular graphs from natural language descriptions. We believe that our model would have a broad impact on AI-empowered fields across disciplines such as biology, chemistry, materials, environment, and medicine, among others.

From <https://arxiv.org/abs/2209.05481> 


[Submitted on 29 May 2023]
ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback
Shengchao Liu, Jiongxiao Wang, Yijin Yang, Chengpeng Wang, Ling Liu, Hongyu Guo, Chaowei Xiao
	Recent advancements in conversational large language models (LLMs), such as ChatGPT, have demonstrated remarkable promise in various domains, including drug discovery. However, existing works mainly focus on investigating the capabilities of conversational LLMs on chemical reaction and retrosynthesis. While drug editing, a critical task in the drug discovery pipeline, remains largely unexplored. To bridge this gap, we propose ChatDrug, a framework to facilitate the systematic investigation of drug editing using LLMs. ChatDrug jointly leverages a prompt module, a retrieval and domain feedback (ReDF) module, and a conversation module to streamline effective drug editing. We empirically show that ChatDrug reaches the best performance on 33 out of 39 drug editing tasks, encompassing small molecules, peptides, and proteins. We further demonstrate, through 10 case studies, that ChatDrug can successfully identify the key substructures (e.g., the molecule functional groups, peptide motifs, and protein structures) for manipulation, generating diverse and valid suggestions for drug editing. Promisingly, we also show that ChatDrug can offer insightful explanations from a domain-specific perspective, enhancing interpretability and enabling informed decision-making. This research sheds light on the potential of ChatGPT and conversational LLMs for drug editing. It paves the way for a more efficient and collaborative drug discovery pipeline, contributing to the advancement of pharmaceutical research and development.

From <https://arxiv.org/abs/2305.18090> 

[Submitted on 13 Jun 2023]
Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models
Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, Huajun Chen
	Large Language Models (LLMs), with their remarkable task-handling capabilities and innovative outputs, have catalyzed significant advancements across a spectrum of fields. However, their proficiency within specialized domains such as biomolecular studies remains limited. To address this challenge, we introduce Mol-Instructions, a meticulously curated, comprehensive instruction dataset expressly designed for the biomolecular realm. Mol-Instructions is composed of three pivotal components: molecule-oriented instructions, protein-oriented instructions, and biomolecular text instructions, each curated to enhance the understanding and prediction capabilities of LLMs concerning biomolecular features and behaviors. Through extensive instruction tuning experiments on the representative LLM, we underscore the potency of Mol-Instructions to enhance the adaptability and cognitive acuity of large models within the complex sphere of biomolecular studies, thereby promoting advancements in the biomolecular research community. Mol-Instructions is made publicly accessible for future research endeavors and will be subjected to continual updates for enhanced applicability.

From <https://arxiv.org/abs/2306.08018> 


GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning
Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, Qi Liu
Abstract
Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs. To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset consisting of more than two thousand molecule tasks with corresponding instructions derived from task descriptions. We pretrain GIMLET on the molecule tasks along with instructions, enabling the model to transfer effectively to a broad range of tasks. Experimental results demonstrate that GIMLET significantly outperforms molecule-text baselines in instruction-based zero-shot learning, even achieving closed results to supervised GNN models on tasks such as toxcast and muv.

From <https://www.biorxiv.org/content/10.1101/2023.05.30.542904v2.abstract> 


[Submitted on 18 May 2023 (v1), last revised 26 May 2023 (this version, v2)]
MolXPT: Wrapping Molecules with Text for Generative Pre-training
Zequn Liu, Wei Zhang, Yingce Xia, Lijun Wu, Shufang Xie, Tao Qin, Ming Zhang, Tie-Yan Liu
	Generative pre-trained Transformer (GPT) has demonstrates its great success in natural language processing and related techniques have been adapted into molecular modeling. Considering that text is the most important record for scientific discovery, in this paper, we propose MolXPT, a unified language model of text and molecules pre-trained on SMILES (a sequence representation of molecules) wrapped by text. Briefly, we detect the molecule names in each sequence and replace them to the corresponding SMILES. In this way, the SMILES could leverage the information from surrounding text, and vice versa. The above wrapped sequences, text sequences from PubMed and SMILES sequences from PubChem are all fed into a language model for pre-training. Experimental results demonstrate that MolXPT outperforms strong baselines of molecular property prediction on MoleculeNet, performs comparably to the best model in text-molecule translation while using less than half of its parameters, and enables zero-shot molecular generation without finetuning.

From <https://arxiv.org/abs/2305.10688> 


[Submitted on 29 Jan 2023]
Unifying Molecular and Textual Representations via Multi-task Language Modelling
Dimitrios Christofidellis, Giorgio Giannone, Jannis Born, Ole Winther, Teodoro Laino, Matteo Manica
	The recent advances in neural language models have also been successfully applied to the field of chemistry, offering generative solutions for classical problems in molecular design and synthesis planning. These new methods have the potential to optimize laboratory operations and fuel a new era of data-driven automation in scientific discovery. However, specialized models are still typically required for each task, leading to the need for problem-specific fine-tuning and neglecting task interrelations. The main obstacle in this field is the lack of a unified representation between natural language and chemical representations, complicating and limiting human-machine interaction. Here, we propose a multi-domain, multi-task language model to solve a wide range of tasks in both the chemical and natural language domains. By leveraging multi-task learning, our model can handle chemical and natural language concurrently, without requiring expensive pre-training on single domains or task-specific models. Interestingly, sharing weights across domains remarkably improves our model when benchmarked against state-of-the-art baselines on single-domain and cross-domain tasks. In particular, sharing information across domains and tasks gives rise to large improvements in cross-domain tasks, the magnitude of which increase with scale, as measured by more than a dozen of relevant metrics. Our work suggests that such models can robustly and efficiently accelerate discovery in physical sciences by superseding problem-specific fine-tuning and enhancing human-model interactions.

From <https://arxiv.org/abs/2301.12586> 





[Submitted on 17 Apr 2023]
Empowering AI drug discovery with explicit and implicit knowledge
Yizhen Luo, Kui Huang, Massimo Hong, Kai Yang, Jiahuan Zhang, Yushuai Wu, Zaiqin Nie
	Motivation: Recently, research on independently utilizing either explicit knowledge from knowledge graphs or implicit knowledge from biomedical literature for AI drug discovery has been growing rapidly. These approaches have greatly improved the prediction accuracy of AI models on multiple downstream tasks. However, integrating explicit and implicit knowledge independently hinders their understanding of molecules. Results: We propose DeepEIK, a unified deep learning framework that incorporates both explicit and implicit knowledge for AI drug discovery. We adopt feature fusion to process the multi-modal inputs, and leverage the attention mechanism to denoise the text information. Experiments show that DeepEIK significantly outperforms state-of-the-art methods on crucial tasks in AI drug discovery including drug-target interaction prediction, drug property prediction and protein-protein interaction prediction. Further studies show that benefiting from explicit and implicit knowledge, our framework achieves a deeper understanding of molecules and shows promising potential in facilitating drug discovery applications.

From <https://arxiv.org/abs/2305.01523> 


[Submitted on 14 Feb 2023]
PrefixMol: Target- and Chemistry-aware Molecule Design via Prefix Embedding
Zhangyang Gao, Yuqi Hu, Cheng Tan, Stan Z. Li
	Is there a unified model for generating molecules considering different conditions, such as binding pockets and chemical properties? Although target-aware generative models have made significant advances in drug design, they do not consider chemistry conditions and cannot guarantee the desired chemical properties. Unfortunately, merging the target-aware and chemical-aware models into a unified model to meet customized requirements may lead to the problem of negative transfer. Inspired by the success of multi-task learning in the NLP area, we use prefix embeddings to provide a novel generative model that considers both the targeted pocket's circumstances and a variety of chemical properties. All conditional information is represented as learnable features, which the generative model subsequently employs as a contextual prompt. Experiments show that our model exhibits good controllability in both single and multi-conditional molecular generation. The controllability enables us to outperform previous structure-based drug design methods. More interestingly, we open up the attention mechanism and reveal coupling relationships between conditions, providing guidance for multi-conditional molecule generation.

From <https://arxiv.org/abs/2302.07120> 

[Submitted on 28 Jan 2023]
ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts
Minghao Xu, Xinyu Yuan, Santiago Miret, Jian Tang
	Current protein language models (PLMs) learn protein representations mainly based on their sequences, thereby well capturing co-evolutionary information, but they are unable to explicitly acquire protein functions, which is the end goal of protein representation learning. Fortunately, for many proteins, their textual property descriptions are available, where their various functions are also described. Motivated by this fact, we first build the ProtDescribe dataset to augment protein sequences with text descriptions of their functions and other important properties. Based on this dataset, we propose the ProtST framework to enhance Protein Sequence pre-training and understanding by biomedical Texts. During pre-training, we design three types of tasks, i.e., unimodal mask prediction, multimodal representation alignment and multimodal mask prediction, to enhance a PLM with protein property information with different granularities and, at the same time, preserve the PLM's original representation power. On downstream tasks, ProtST enables both supervised learning and zero-shot prediction. We verify the superiority of ProtST-induced PLMs over previous ones on diverse representation learning benchmarks. Under the zero-shot setting, we show the effectiveness of ProtST on zero-shot protein classification, and ProtST also enables functional protein retrieval from a large-scale database without any function annotation.

From <https://arxiv.org/abs/2301.12040> 




[Submitted on 29 Sep 2022]
Improving Molecular Pretraining with Complementary Featurizations
Yanqiao Zhu, Dingshuo Chen, Yuanqi Du, Yingze Wang, Qiang Liu, Shu Wu
	Molecular pretraining, which learns molecular representations over massive unlabeled data, has become a prominent paradigm to solve a variety of tasks in computational chemistry and drug discovery. Recently, prosperous progress has been made in molecular pretraining with different molecular featurizations, including 1D SMILES strings, 2D graphs, and 3D geometries. However, the role of molecular featurizations with their corresponding neural architectures in molecular pretraining remains largely unexamined. In this paper, through two case studies -- chirality classification and aromatic ring counting -- we first demonstrate that different featurization techniques convey chemical information differently. In light of this observation, we propose a simple and effective MOlecular pretraining framework with COmplementary featurizations (MOCO). MOCO comprehensively leverages multiple featurizations that complement each other and outperforms existing state-of-the-art models that solely relies on one or two featurizations on a wide range of molecular property prediction tasks.

From <https://arxiv.org/abs/2209.15101> 


Translation between Molecules and Natural Language
Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Heng Ji
	Joint representations between images and text have been deeply investigated in the literature. In computer vision, the benefits of incorporating natural language have become clear for enabling semantic-level control of images. In this work, we present MolT5?a self-supervised learning framework for pretraining models on a vast amount of unlabeled natural language text and molecule strings. MolT5 allows for new, useful, and challenging analogs of traditional vision-language tasks, such as molecule captioning and text-based de novo molecule generation (altogether: translation between molecules and language), which we explore for the first time. Furthermore, since MolT5 pretrains models on single-modal data, it helps overcome the chemistry domain shortcoming of data scarcity. Additionally, we consider several metrics, including a new cross-modal embedding-based metric, to evaluate the tasks of molecule captioning and text-based molecule generation. By interfacing molecules with natural language, we enable a higher semantic level of control over molecule discovery and understanding--a critical task for scientific domains such as drug discovery and material design. Our results show that MolT5-based models are able to generate outputs, both molecule and text, which in many cases are high quality and match the input modality. On molecule generation, our best model achieves 30% exact matching test accuracy (i.e., it generates the correct structure for about one-third of the captions in our held-out test set).

From <https://arxiv.org/abs/2204.11817#> 


A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals
Zheni Zeng # 1, Yuan Yao # 1, Zhiyuan Liu 2, Maosong Sun 3
Affiliations expand
• PMID: 35165275
 
• PMCID: PMC8844428
 
• DOI: 10.1038/s41467-022-28494-3
Free PMC article
Full text linksCite
Abstract
To accelerate biomedical research process, deep-learning systems are developed to automatically acquire knowledge about molecule entities by reading large-scale biomedical data. Inspired by humans that learn deep molecule knowledge from versatile reading on both molecule structure and biomedical text information, we propose a knowledgeable machine reading system that bridges both types of information in a unified deep-learning framework for comprehensive biomedical research assistance. We solve the problem that existing machine reading models can only process different types of data separately, and thus achieve a comprehensive and thorough understanding of molecule entities. By grasping meta-knowledge in an unsupervised fashion within and across different information sources, our system can facilitate various real-world biomedical applications, including molecular property prediction, biomedical relation extraction and so on. Experimental results show that our system even surpasses human professionals in the capability of molecular property comprehension, and also reveal its promising potential in facilitating automatic drug discovery and documentation in the future.

From <https://pubmed.ncbi.nlm.nih.gov/35165275/> 


BioassayCLR: Prediction of biological activity for novel bioassays based on rich textual descriptions 
Andreu Vall, Sepp Hochreiter, Günter Klambauer Institute for Machine Learning Johannes Kepler University Linz, Austria klambauer@ml.jku.at Abstract Screening molecules for desired biological activities with bioassays is at the core of the drug discovery process. The data produced by bioassays enable building quantitative structure-activity relationship (QSAR) models that are fundamental components of computer-aided drug discovery. Despite the advances brought by Deep Learning-based QSAR models, it is still unclear how to build these models for new bioassays for which no active nor inactive molecules are known. To ameliorate this problem, we propose BioassayCLR, a machine learning method that leverages rich textual bioassay descriptions for modeling. Our model takes as input both the chemical structure of a molecule and the textual description of the bioassay and outputs the predicted activity for this pair. The approach can be viewed as a contrastive learning approach in which representations of both molecules and bioassays should be learned, which are similar if the molecule-bioassay pair is active and dissimilar if the pair is inactive. We perform experiments on bioassay descriptions and molecules from PubChem with 223,219,241 records of molecule-bioassay activity, corresponding to 2,120,811 unique molecules and 21,002 unique bioassays. On a strict temporal hold-out set with 615 unseen bioassays and 248,290 unseen molecules, BioassayCLR reaches an AUROC of 63.97 ± 0.47 outperforming the baselines using simple textual similarity by a margin, whereas all other QSAR methods yield random performance of 50.00. To our knowledge, this is the first time that a textual representation of a bioassay is directly fed into a QSAR model and, thus, the first method that can produce accurate predictions for bioassays that are only described by natural language. Because of these properties, our method allows for zero-shot transfer learning in drug discovery. 

Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries
Carl Edwards, ChengXiang Zhai, Heng Ji

Abstract
We propose a new task, Text2Mol, to retrieve molecules using natural language descriptions as queries. Natural language and molecules encode information in very different ways, which leads to the exciting but challenging problem of integrating these two very different modalities. Although some work has been done on text-based retrieval and structure-based retrieval, this new task requires integrating molecules and natural language more directly. Moreover, this can be viewed as an especially challenging cross-lingual retrieval problem by considering the molecules as a language with a very unique grammar. We construct a paired dataset of molecules and their corresponding text descriptions, which we use to learn an aligned common semantic embedding space for retrieval. We extend this to create a cross-modal attention-based model for explainability and reranking by interpreting the attentions as association rules. We also employ an ensemble approach to integrate our different architectures, which significantly improves results from 0.372 to 0.499 MRR. This new multimodal approach opens a new perspective on solving problems in chemistry literature understanding and molecular machine learning.

From <https://aclanthology.org/2021.emnlp-main.47/> 

Automated extraction of chemical synthesis actions from experimental procedures
• Alain C. Vaucher, 
• Federico Zipoli, 
• Joppe Geluykens, 
• Vishnu H. Nair, 
• Philippe Schwaller & 
• Teodoro Laino 
Abstract
Experimental procedures for chemical synthesis are commonly reported in prose in patents or in the scientific literature. The extraction of the details necessary to reproduce and validate a synthesis in a chemical laboratory is often a tedious task requiring extensive human intervention. We present a method to convert unstructured experimental procedures written in English to structured synthetic steps (action sequences) reflecting all the operations needed to successfully conduct the corresponding chemical reactions. To achieve this, we design a set of synthesis actions with predefined properties and a deep-learning sequence to sequence model based on the transformer architecture to convert experimental procedures to action sequences. The model is pretrained on vast amounts of data generated automatically with a custom rule-based natural language processing approach and refined on manually annotated samples. Predictions on our test set result in a perfect (100%) match of the action sequence for 60.8% of sentences, a 90% match for 71.3% of sentences, and a 75% match for 82.4% of sentences.
From <https://www.nature.com/articles/s41467-020-17266-6> 
