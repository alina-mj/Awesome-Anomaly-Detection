# Awesome Abnormaly Detection

## Out-of-Distribution (OOD) 
  - ["A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"](https://arxiv.org/pdf/1610.02136.pdf) (ICLR 2017)
    - Detect OOD samples by predicting softmax class probability, but some OOD samples show overconfident class score.
    - Propose OOD task first and propose the criteria of it.
  - ["Training Confidence-Calibrated Classifiers for Detecting Out-oF-Distribution Samples"](https://arxiv.org/pdf/1711.09325.pdf) (ICLR 2018)
    - Use modified GAN
    - Generator: generate boundary OOD samples that appear to be at the boundary of in-distribution data. \
      Classifier: assign OOD samples uniform class probabilities.
  - ["Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks"](https://arxiv.org/pdf/1706.02690.pdf) (ODIN) (ICLR 2018) 
    - Temperature scaling: to push the softmax scores of ID and OOD samples further apart from each other, just for test (T=1000)
    - Input preprocessing: add small perturbations to the input to increase the maximum softmax score (The increase on ID samples are greater than those on OOD samples)
  - ["Deep Anomaly Detection with Outlier Exposure"](https://arxiv.org/pdf/1812.04606.pdf) (ICLR 2019)
    - Use auxiliary dataset of outliers
