# Deep Learning Architectures

## Overview
In addition to mainstream models such as the ResNet family and Transformer models, this study also explores the use of more complex architectures, namely hybrid models. In real-world applications, there might exist sequences of frames, defined as a series of three or more consecutive frames, and stills (i.e., unrelated individual frames). The proposed dual-head architecture is created to handle both situations. For the still head, each frame is passed through the CNN models, and global average pooling (GAP) performs a downsampling operation. The classifier has fully-connected layers (FC_cls) will output each feature into a frame logit. This process will be repeated for each frame, and at the end, all of the logits will be aggregated and averaged to obtain the still-level logit. For the sequence head, the processing of each frame at the beginning is similar to that of the still head. This process is repeated for each sequence, and the features are aggregated as input to the LSTM module or STM .

![Hybrid Architecture](assets/Training&20Architecture.png)

## ResNet-50
Figure 1 shows the complete architecture of the ResNet-50 model. ResNet-50 is one of the most popular deep learning models for image classification. The model used 48 convolutional layers, forming 16 residual blocks, 1 max-pooling layer, and 1 average-pooling layer. The structure of ResNet-50 allows the model to effectively capture and propagate complex features across different layers (Xue et al., 2024).

## Swin Transformer V2-Base
Figure 2 shows the architecture of the Swin Transformer model, consisting of four stages. First, an image is split into non-overlapping patches and tokenized. The feature dimension of each patch is 48. A linear embedding layer is applied on this feature to project it to an arbitrary dimension (denoted as C). At the first stage, Swin Transformer blocks are applied to the patch tokens, maintaining the size of H/4×W/4. To produce a hierarchical representation, the number of tokens is reduced by patch merging layers. At the second stage, Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at H/8×W/8. The procedure is repeated for stages 3 and 4 with output resolutions of H/16×W/16 and H/32×W/32, respectively. At last, the feature is passed into the fully-connected layers FC_cls to produce the classification output (Liu et al., 2021).

## CNN-LSTM
LSTM, whose architecture is presented in Figure 2 12, is a type of recurrent neural network that uses gating functions to minimize the exploding and vanishing gradients problem (Sainju & Jiang, 2020). Each LSTM unit has a cell state (c_i) and three different gates: the input gate, the output gate, and the forget gate. The forget gate (f_i )  determines how much information from a previous cell state to “forget.” The input gate (i_i )  determines the amount of contribution an input feature vector makes to the current cell state. Last, the output gate (o_i )  decides which current LSTM unit is going to the output based on the cell states. The current hidden state (h_i )  resulting from the modification is output and passed to the next LSTM unit. 
For this study, each sequence feature is passed through the LSTM operation to obtain the hidden state (h_i ). Finally, all hidden states will be aggregated and averaged; the result will then be passed through the FC_cls to obtain the sequence-level logit. 

## CNN-STM
In the STM, the sequences are tokenized, positionally embedded, and passed into the STM, which consists of L encoders, as seen in Figure 2 13 (Shahid et al., 2025). Each encoder layer comprises multi-head attention and a multi-layer perceptron (MLP). The normalization layer is used before each multi-head attention and MLP in the encoder layer. Afterwards, the encoded features will be pooled using the class token. Similar to the LSTM module, the result will be passed to the FC_cls to obtain the sequence-level logit.

At last, the weighted logit that fuses the sequence and still logits is calculated based on Equation 7. The argmax function will return the classification output with the highest weighted logit.

logit_weighted=w_seq logit_seq+w_still logit_still	

Where w_seq  is the weight of the sequence logit, and w_still  is the weight of the still logit.

