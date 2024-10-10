# QT-TDM
<img src="https://github.com/2M-kotb/QT-TDM/blob/main/QT-TDM.png" width=100% height=100%>


The official Pytorch implementation of QT-TDM model presented in paper:

[QT-TDM:Planning with Transformer Dynamics Model and Autoregressive Q-Learning](https://arxiv.org/pdf/2407.18841)


# Method
QT-TDM is a Transformer-based model-based algorithm that consists of two modules: 

__(a)Transformer Dynamics Model (TDM).__  

__(b) Q-Transformer (QT).__

While Transformers serve as large, expressive, and robust dynamics models, they are not optimized for fast real-time planning due to the autoregressive token predictions and the per-dimension tokenization scheme.
QT-TDM improves real-time planning capabilities and efficiency by shortenning the planning horizon and utilizing the Q-Transformer model to estimate a long-term return beyond the short-term planning horizon, as shown in the following Fig. :

<img src="https://github.com/2M-kotb/QT-TDM/blob/main/planning.png" width=45% height=45%>
