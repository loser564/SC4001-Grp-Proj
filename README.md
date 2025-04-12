**Introduction**

Deep learning has transformed how we tackle image recognition, making it possible to classify even complex images with high accuracy. In this project, we aim to classify 102 flower types using the Oxford Flowers 102 dataset, a challenging dataset with significant visual variation and similarities among classes. The goal of this project is to build a deep learning model that accurately classifies each flower type in this dataset. We experiment with popular models like ResNet, VGG and DeiT, using transfer learning to take advantage of pretrained models. Our goal is to build a robust classifier that demonstrates deep learning’s potential for detailed image classification.

**Data Preparation**

For the data cleaning process, we started with two files provided in .mat format: Imagelabel.mat and setid.mat. Since these were MATLAB files, we needed to use the loadmat function to load them into Python. The imagelabels file contained the label data for each image, while setid included information about how the images were divided into training, validation, and testing sets.

To work with the labels effectively, we converted the imagelabels data into a pandas DataFrame. This allowed us to map each image ID to its corresponding class. For the setid file, we created three separate DataFrames, one each for training, validation, and testing. 

Next, we associate image filenames with their image IDs. Finally, we used pandas' merge function to bring everything together into a new DataFrame called df\_image\_labels

To prepare the dataset for training, we needed to organise the images into folders based on their classes. This meant creating separate folders for each class within each dataset type (training, validation, testing)

In order to do this, we used a for loop and the shutil package to automate this sorting process. We loop through each row in the dataframe and move each image to its corresponding folder according to its type and class. 

\


**Data Augmentation**

To make the training data more varied and help prevent overfitting, we used a few data augmentation techniques:

- Horizontal Flip: Randomly flips images to give variety in orientation.

- Rotation: Rotates images up to 20 degrees to add variation in positioning.

We also resize all images to 224 x 224 pixels to match the input size expected by most neural networks. This resizing keeps the image dimensions consistent for batch processing. Each image is converted from a PIL image to a PyTorch tensor and then normalised to match the ImageNet dataset. Since ResNet18 was pre trained on ImageNet, this normalisation makes our images similar to what the model was originally trained on which helps it to perform better.

\


**Mixup Augmentation**

We added another technique called MixUp to improve generalisation of deep learning models. MixUp generates new training samples by mixing two random samples from the batch and their corresponding labels, creating a linear combination of both images and labels. Here’s a breakdown of how the code works:

\


**Training Parameters**

- For all models, we used the same base function for training

  - 2 phases, training first then validation to allow an all in one functionality increasing code readability

  - Application of mixup in training phase

  - Calculation of loss with mixup

\


**Experiment**

**ResNet**

Residual Networks (ResNet) is a CNN-based architecture that introduces residual connections, which helps mitigate the vanishing gradient problem in deep learning networks. The residual connection is a shortcut that allows the information to bypass one or more layers in the network and reach the output directly. It allows the network to learn the residual function and make small updates to the parameters, which enables the network to converge faster and achieve better performance (Banjara, 2024).

We started with ResNet50, but the 50-layer depth led to a vanishing gradient problem, meaning that the gradients became too small since they are back-propagated through the network with many layers, making it difficult for the model to effectively learn and update weights, particularly when the dataset is smaller. 

To solve this, we switched to ResNet18, a simpler version of ResNet with 18 layers. For this model, we initially tested with 0 frozen layers, allowing all layers to be trained to fit our dataset. However, this led to high training accuracy and poor validation accuracy, which indicated that the model was overfitting. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdoWkcibRnZTqWvLLl5b-d2Goq0F9p13383cuAeO4veV1FuOqJqZ0RPPeP2C6joZnLyz-RB3Q8VTlJ-NZ5WXwOD6v_e_LuTPYOYhQwIF20QOelj_v9dxto_H9r7Dyocmm7aWHxqOgn5Cqd1AmjVO4m4MatT?key=daAkxYBlLAOm-2zCGjhQQg)

To address this, we froze the initial layers of the model. Freezing layers limits the number of trainable parameters, allowing the model to retain its general features from pretraining and reducing the risk of overfitting. Other hyperparameter settings include the choice of 20 epochs with early stopping patience of 5, as we noticed that the model does not show any significant improvement after around 10 epochs for the 0 frozen layers case. This ensured that we gave enough buffer for the different layers to train and prevented overfitting. 

We started with a learning rate(lr) of 0.001, and implemented a learning rate scheduler such that the learning rate decreases by 0.1 (new\_lr = lr x 0.1) every 7 epochs, this ensures finer adjustments as the model approaches convergence and prevents the model from overshooting the optimal solution. 

The optimizer of choice was Adam, an adaptive optimizer that adjusts learning rate based on first and second moments of gradients. This works well for fine-tuning pretrained models because it is less sensitive to initial learning rate and converges faster. 

After testing by freezing 0 to 5 layers, we found the best validation accuracy (0.7422) came from freezing the first five layers.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdUZZPZEktewT599LHubBE1Ie84E0V8w41t8-LOIfQFiZDwpnXEwOdgSbuIErp3E24G7VVgOL6df_tgalI6f4G1H73ayUesyB4hUf2KjwObhxkv6V5xvzdYCb3c8-64lXjA_gr1zA?key=daAkxYBlLAOm-2zCGjhQQg)

\
\
\


**VGG16**

Visual Geometry Group (VGG) is also a CNN-based architecture. The VGG architecture maintains a consistent arrangement of convolutional (with 3x3 filter) and max-pooling layers throughout, allowing it to effectively capture hierarchical features. At the end of the network, two fully connected layers are used to consolidate these features, followed by a soft max layer for output classification. Similarly to ResNet18, it was trained on ImageNet.

For this model, we switched to using an SGD optimizer instead of Adam because VGG16 is a larger pre-trained model than ResNet18, and hence SGD with momentum encourages a stable and flatter optimization path compared to Adam. Furthermore VGG16 includes batch normalization layers which works well with SGD while Adam might interfere with the internal scaling dynamics of batch normalization.

The learning rate scheduler remained the same, but we increased the maximum number of epochs to 30, taking into account the increased depth of the model, but kept early stopping with patience of 5 to prevent overtraining.

We initially froze 10 layers, but the validation accuracy was low at less than 0.60.

 ![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdr4kKxRPCljfSHZ7JpzMqlniaRBiUbY5L4bhXpbC7tNerx-kjGBGJkC511CgBBeHLkMH2Q-oaLJSm15Q0ltu-dMZDLmNDjfEbKTBFDkJiPGyIR_4S44UVStDaoC7PLheeSnbgII6nvNzV6w8Me1Xt2j-2T?key=daAkxYBlLAOm-2zCGjhQQg)

After experimenting with different numbers of frozen layers, we discovered that freezing 8 layers gave us one of the best validation accuracy for this model, 13 gave a slightly higher accuracy, but 8 frozen layers would give the model more fine-tuning and make it more specific to the dataset.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeWtbFmZbNPfZI7YN10Y8ZBfM4rVfhrDaPnl5gWPpsFYVcCDnAsCYiQZxx-EPaStVquxI3pWGf3nScrZ5lnY4rgF4j02ZCP731SQYTIZBqLuSxzLAhijL2AUFh3BcKMr0PQo1bE?key=daAkxYBlLAOm-2zCGjhQQg)

\


**VGG19**

For VGG19, we used a similar hyperparameter set up as with VGG16. Similarly to VGG16, we tested the validation accuracies of freezing a range of layers to determine the optimal layers to be frozen. We achieved the best validation accuracy by freezing 13 layers.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeC0DphIY5gDXVontCAjeq6n3LcDyfaaSHQM4O5-WnPrxBk517Yp9BFBMJ-YHtTY-Uwc1xEFVlb0SmC394d6v3MstjtA-4jkd0XHNWs3ohozviRH4S3DYeSLf3I-wbquyRoXPdm?key=daAkxYBlLAOm-2zCGjhQQg)

**DeiT**

Data-Efficient Image Transformer (DeiT) is a transformer-based architecture designed to achieve high performance in image classification tasks with significantly less data compared to traditional vision transformers using knowledge distillation and strong data augmentation.

We used the same maximum epochs, early stopping patience and learning rate as the VGG model.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdV-hUvbm7BwUf3XhMMDEgZhl3qbB47iZXyFYaEGhoB81aek59eVhNgZHkkoOYPJ3IKwzYmnqiMjyE1c3AOQNSHQcjX_-oNyra8c7j0q3d2_0ay1VyI0BYXTh9Edl3Y-eFLGLmxpQ?key=daAkxYBlLAOm-2zCGjhQQg)

**Soft Voting Ensemble**

To further improve prediction accuracy, we implemented the soft voting ensemble method. Soft voting combines the probabilities of individual models to make a prediction. Each model provides the probabilities, and the ensemble will aggregate them, predicting the class with the highest average probabilities.

For our experiment, we made use of the above 4 models - ResNet18, VGG16, VGG19 and DeiT, using different combinations for the ensemble. This will be discussed in the next section.

\


**Result+Analysis**

**Individual Models**

|                            |                   |
| -------------------------- | ----------------- |
| **Model**                  | **Test Accuracy** |
| ResNet18 (5 Frozen Layers) | 64.31%            |
| VGG16 (8 Frozen Layers)    | 73.23%            |
| VGG19 (13 Frozen Layers)   | 71.05%            |
| DeiT                       | 80.37%            |

The significantly lower test accuracy for ResNet18 indicates the model struggles with the complexity of the classification since it is more suited for tasks where computational efficiency is a priority (1), which is a trade-off to maximising accuracy by learning more details. 

VGG models have a better accuracy which could be due to the deeper architectures compared to ResNet (2), allowing it to capture and differentiate more complex patterns in the images. VGG16 performed better than VGG19, indicating that the additional 3 layers in VGG19 may have caused overfitting to occur, and the model cannot generalise on unseen data as well.

DeiT significantly outperforms the other models which could be attributed to it being suitable for smaller training datasets (3), like ours with 10 images per class (1020 images total). The transformer-based approach is likely better suited for capturing complex dependencies and relationships within the images, which may be critical in distinguishing between 102 classes.

\


**Soft Voting Ensemble Method**

|         |                              |                   |
| ------- | ---------------------------- | ----------------- |
|         | **Combinations**             | **Test Accuracy** |
| Model X | VGG16, VGG19, ResNet18       | 76.66%            |
| Model Y | VGG16, VGG19, DieT           | 80.01%            |
| Model Z | VGG16, VGG19, ResNet18, DeiT | 80.71%            |

Overall, there is improvement in the test accuracies for the ensembles compared to their respective individual models by themselves. This could be attributed to the fact that all 4 models have different architectures and may be more confident in predicting certain classes over others. Taking the average probabilities allows the ensemble model to cover wider ranges of classes, which is especially important in our case with 102 classes.

For Model X, it leverages on the strengths of convolutional networks, which excel at capturing localised features. CNNs treat images as structured pixel arrays and process them with convolutions. Through these trainable convolutional filters, CNNs create feature maps which are representations of the original image. This means that the convolution operation affects only a small patch of an image at a time, and can be considered a local operation (4). The lack of a transformer-based model, such as DeiT, might limit its ability to capture complex global patterns.

For Model Y, the accuracy is much higher after replacing ResNet18 with DieT. DieT, being a transformer, operates on a more global scale (4). This allows the ensemble method to capture both smaller details of features and global pattern recognition, boosting the performance of the ensemble.

For Model Z, we observe the addition of ResNet18 to the previous ensemble results in minimal improvement in accuracy (increase of 0.02%). This suggests that ResNet18 adds little value to the ensemble and can be excluded from the ensemble. Hence, we prefer  Model Y as it is simpler than Model Z.

\


**Conclusion**

In this study, we explored various models and ensemble techniques to classify images into 102 distinct categories for flower image classification. By examining the performance of individual models such as ResNet18, VGG16, VGG19, and DeiT, we observed that transformer-based models like DeiT outperformed convolutional networks on our relatively small dataset. Soft voting ensemble strategy combined the strengths of both convolutional networks and transformers, allowing for learning on both fine-grained details of images and broader contextual information, achieving improved results over individual models. Overall, we chose the ensemble of VGG16, VGG19 and DeiT (Model Y) to be the best combination.

Future work could explore additional transformer methods or ensemble methods to push performance even further.

**References**

<https://www.productteacher.com/quick-product-tips/resnet18-and-resnet50#:~:text=ResNet18%20consists%20of%2018%20layers,where%20computational%20resources%20are%20limited>.

<https://www.researchgate.net/publication/369224507_Comparison_of_VGG-16_VGG-19_and_ResNet-101_CNN_Models_for_the_purpose_of_Suspicious_Activity_Detection>

<https://medium.com/@zakhtar2020/deit-data-efficient-image-transformer-overview-acd1cb3b1dcf>  

<https://www.picsellia.com/post/are-transformers-replacing-cnns-in-object-detection#5-How-and-Why-Are-ViTs-Different-From-CNNs->

<https://www.analyticsvidhya.com/blog/2023/02/deep-residual-learning-for-image-recognition-resnet-explained/>
