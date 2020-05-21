# Awesome-Anomaly-Detection
## Contents
  1. Image \
    (1) Out-of-Distribution (OOD) Detection \
    (2) One-Class Classification: Novelty Detection, Outlier Detection 
  2. Video
  3. Time-Series \
    (1) Multivariate \
    (2) Univariate
  4. Graph
  + [Defect Detection](https://github.com/alina-mj/Awesome-Defect-Detection.git)



## 0. Survey for Overall Anomaly Detection
- Chalapathy, Raghavendra, and Sanjay Chawla. "Deep learning for anomaly detection: A survey." arXiv preprint arXiv:1901.03407 (2019) | [paper](https://arxiv.org/pdf/1901.03407.pdf)
- anomaly-detection-resources | [git](https://github.com/yzhao062/anomaly-detection-resources.git)
- awesome-anomaly-detection | [git](https://github.com/hoya012/awesome-anomaly-detection.git)


## 1. Image
### (1) Out-of-Distribution (OOD) Detection
- Hendrycks, Dan, and Kevin Gimpel. "A baseline for detecting misclassified and out-of-distribution examples in neural networks." ICLR 2017 | [paper](https://arxiv.org/pdf/1610.02136.pdf) | [git](https://github.com/hendrycks/error-detection.git)
- Lee, Kimin, et al. "Training confidence-calibrated classifiers for detecting out-of-distribution samples." ICLR 2018 | [paper](https://arxiv.org/pdf/1711.09325.pdf) | [git](https://github.com/alinlab/Confident_classifier.git)
- Liang, Shiyu, Yixuan Li, and Rayadurgam Srikant. "Enhancing the reliability of out-of-distribution image detection in neural networks." ICLR 2018 | [paper](https://arxiv.org/pdf/1706.02690.pdf) | [git](https://github.com/facebookresearch/odin.git)
- Lee, Kimin, et al. "A simple unified framework for detecting out-of-distribution samples and adversarial attacks." NeurlPS 2018 | [paper](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks.pdf) | [git](https://github.com/pokaxpoka/deep_Mahalanobis_detector.git)
- Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich. "Deep anomaly detection with outlier exposure." ICLR 2019 | [paper](https://arxiv.org/pdf/1812.04606.pdf) | [git](https://github.com/hendrycks/outlier-exposure.git)

- DeVries, Terrance, and Graham W. Taylor. "Learning confidence for out-of-distribution detection in neural networks." arXiv preprint arXiv:1802.04865 (2018) | [paper](https://arxiv.org/pdf/1802.04865.pdf) | [git](https://github.com/uoguelph-mlrg/confidence_estimation.git)
- Yu, Qing, and Kiyoharu Aizawa. "Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy." ICCV 2019 | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Unsupervised_Out-of-Distribution_Detection_by_Maximum_Classifier_Discrepancy_ICCV_2019_paper.pdf)
- Nalisnick, Eric, et al. "Do deep generative models know what they don't know?." ICLR 2017 | [paper](https://arxiv.org/pdf/1810.09136.pdf)
- Vyas, Apoorv, et al. "Out-of-distribution detection using an ensemble of self supervised leave-out classifiers." ECCV 2018 | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Apoorv_Vyas_Out-of-Distribution_Detection_Using_ECCV_2018_paper.pdf)
- Choi, Hyunsun, Eric Jang, and Alexander A. Alemi. "Waic, but why? generative ensembles for robust anomaly detection." arXiv preprint arXiv:1810.01392 (2018) | [paper](https://arxiv.org/pdf/1810.01392.pdf)
- Serrà, Joan, et al. "Input complexity and out-of-distribution detection with likelihood-based generative models." ICLR 2020 | [paper](https://arxiv.org/pdf/1909.11480.pdf)
- Chen, Jiefeng, et al. "Robust Out-of-distribution Detection in Neural Networks." arXiv preprint arXiv:2003.09711 (2020). Under-review | [paper](https://arxiv.org/pdf/2003.09711.pdf) | [git](https://github.com/jfc43/robust-ood-detection.git)
- Ren, Jie, et al. "Likelihood ratios for out-of-distribution detection." NIPS 2019 | [paper](http://papers.nips.cc/paper/9611-likelihood-ratios-for-out-of-distribution-detection.pdf)
- Hendrycks, Dan, et al. "Using self-supervised learning can improve model robustness and uncertainty." NIPS 2019 | [paper](https://arxiv.org/pdf/1906.12340.pdf) | [git](https://arxiv.org/pdf/1906.12340.pdf)


### (2) One-Class Classification: Novelty Detection, Outlier Detection
#### AE/GAN
- Pidhorskyi, Stanislav, Ranya Almohsen, and Gianfranco Doretto. "Generative probabilistic novelty detection with adversarial autoencoders." NIPS 2018 | [paper](https://arxiv.org/pdf/1807.02588v2.pdf) | [git](https://github.com/podgorskiy/GPND.git)
- Sabokrou, Mohammad, et al. "Adversarially learned one-class classifier for novelty detection." CVPR 2018 | [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf) | [git](https://github.com/khalooei/ALOCC-CVPR2018.git)
- Perera, Pramuditha, Ramesh Nallapati, and Bing Xiang. "Ocgan: One-class novelty detection using gans with constrained latent representations." CVPR 2019 | [paper](https://arxiv.org/pdf/1903.08550.pdf) | [git](https://github.com/PramuPerera/OCGAN.git)
- Ki Hyun Kim, Sangwoo Shim, et al. "RaPP: Novelty Detection with Reconstruction along Projection Pathway." ICLR 2020 | [paper](https://openreview.net/pdf?id=HkgeGeBYDB) | [blog](https://kh-kim.github.io/blog/2020/02/18/rapp.html)
#### Others
- Lukas, et al. "Deep Semi-Supervised Anomaly Detection". ICLR 2020 | [paper](https://openreview.net/pdf?id=HkgH0TEYwH)
- Liron Bergman, and Yedid Hoshen. ”Classification-Based Anomaly Detection for General Data." ICLR 2020 | [paper](https://openreview.net/pdf?id=H1lK_lBtvS)
- Du, Min, Ruoxi Jia, and Dawn Song. "Robust Anomaly Detection and Backdoor Attack Detection Via Differential Privacy." ICLR 2020 | [paper](https://openreview.net/pdf?id=SJx0q1rtvS)



## 2. Video
- [survey] awesome-video-anomaly-detection | [git](https://github.com/fjchange/awesome-video-anomaly-detection.git)
- [survey] Zhu, Sijie, Chen Chen, and Waqas Sultani. "Video Anomaly Detection for Smart Surveillance." arXiv preprint arXiv:2004.00222 (2020) | [paper](https://arxiv.org/pdf/2004.00222.pdf)
- Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection in surveillance videos." CVPR 2018 | [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)
- Liu, Wen, et al. "Future frame prediction for anomaly detection–a new baseline." CVPR 2018 | [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf) | [git](https://github.com/StevenLiuWen/ano_pred_cvpr2018.git)
- Vu, Hung, et al. "Robust anomaly detection in videos using multilevel representations." AAAI 2019 | [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4456)
- Morais, Romero, et al. "Learning regularity in skeleton trajectories for anomaly detection in videos." CVPR 2019 | [paper](https://arxiv.org/pdf/1903.03295.pdf) | [git](https://github.com/RomeroBarata/skeleton_based_anomaly_detection.git)
- Ionescu, Radu Tudor, et al. "Object-centric auto-encoders and dummy anomalies for abnormal event detection in video." CVPR 2019 | [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf)
- Gong, Dong, et al. "Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection." ICCV 2019 | [paper](https://arxiv.org/pdf/1904.02639.pdf) | [git](https://github.com/donggong1/memae-anomaly-detection.git)
- Nguyen, Trong-Nguyen, and Jean Meunier. "Anomaly detection in video sequence with appearance-motion correspondence." ICCV 2019 | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf)



## 3. Time-Series
- [survey] awesome-TS-anomaly-detection | [git](https://github.com/rob-med/awesome-TS-anomaly-detection.git)
- [survey] Blázquez-García, Ane, et al. "A review on outlier/anomaly detection in time series data." arXiv preprint arXiv:2002.04236(2020) | [paper](https://arxiv.org/pdf/2002.04236.pdf)

### (1) Multivariate
- Hundman, Kyle, et al. "Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding." KDD 2018 | [paper](https://arxiv.org/pdf/1802.04431.pdf) | [git](https://github.com/khundman/telemanom.git)
- Ren, Hansheng, et al. "Time-Series Anomaly Detection Service at Microsoft." KDD 2019 | [paper](https://arxiv.org/pdf/1906.03821.pdf)
- Su, Ya, et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." KDD 2019 | [paper](https://dl.acm.org/doi/10.1145/3292500.3330672) | [git](https://github.com/NetManAIOps/OmniAnomaly.git)
- Li, Dan, et al. "Anomaly detection with generative adversarial networks for multivariate time series." 7th International Workshop on Big Data, Streams and Heterogeneous Source Mining: Algorithms, Systems, Programming Models and Applications on KDD 2018 | [paper](https://arxiv.org/abs/1809.04758) | [git](https://github.com/LiDan456/GAN-AD.git)
- Li, Dan, et al. "Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks." International Conference on Artificial Neural Networks. Springer, Cham, 2019 | [paper](https://arxiv.org/pdf/1901.04997v1.pdf) | [git](https://github.com/LiDan456/MAD-GANs.git)
- Zhang, Chuxu, et al. "A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data." AAAI 2019 | [paper](https://arxiv.org/pdf/1811.08055v1.pdf) | [git](https://github.com/7fantasysz/MSCRED.git)
- Gao, Jingkun, et al. "RobustTAD: Robust time series anomaly detection via decomposition and convolutional neural networks." arXiv preprint arXiv:2002.09545 (2020) | [paper](https://arxiv.org/pdf/2002.09545.pdf)
- [code] RNN Time-Series-Anomaly Detection | [git](https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection.git)

- Zong, Bo, et al. "Deep autoencoding gaussian mixture model for unsupervised anomaly detection." ICLR 2018 | [paper](https://openreview.net/forum?id=BJJLHbb0-) | [git1](https://github.com/tnakae/DAGMM.git) | [git2](https://github.com/danieltan07/dagmm)


### (2) Univariate



## 4. Graph
- Bhatia, Siddharth, et al. "MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams." AAAI 2020 | [paper](https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/midas.pdf) | [git](https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/midas.pdf)
