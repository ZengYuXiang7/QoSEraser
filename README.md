# QoSEraser
This is an official PyTorch implementation of paper entitled "QoSEraser: A Data Erasable Framework for Web Service QoS Prediction". 

## Abstract
To select appropriate web services for users, the Quality-of-Service (QoS) based collaborative prediction models are widely used. Despite the success of collaborative prediction models in selecting appropriate web services for users, existing models do not take into account the users' authority to manage their own generated data as stipulated by many privacy-preserving regulations, such as the General Data Protection Regulation (GDPR) and California Consumer Privacy Act (CCPA). Moreover, unlearning is urgently needed due to the security concerns such as data poisoning attacks. Existing QoS prediction methods are not optimized for unlearning, suffering from low model availability when handling unlearning requests by full re-training. 

To address this problem, we propose QoSEraser: a novel efficient machine unlearning framework for QoS prediction tasks. The central concepts of the QoSEraser involve: (1) dividing the training data into multiple shards to train submodels according to the cluster results on graph embeddings induced by random walk on contextual information graph. Such division ensures the preservation of collaborative signals in collected QoS records; (2) a concatenate aggregation method and a stacking & attention-based aggregation method are used to condense information in fragmented embeddings to a uniform one adaptively. Experiments on large-scale datasets show that QoSEraser achieves efficient forgetting learning and outperforms state-of-the-art unlearning approaches in terms of performance.
