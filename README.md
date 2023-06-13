# Automatic Entity Matching

This project is part of our master's program  at ESSEC Business School and CentraleSupélec (Master in Data Science and Business Analytics). We conduct a 6-month project (around 1 day per week) with Société Générale, this corporate research project  aims to bring business values through our skills in data science. 

*Team members: Joshua Fan, Sanjana Gupta, Aline Helburg, Anmol Katiyar, Emma Riguidel*

Our goal is to build an deep learning model that measures similarity of names of two individuals.

Entity matching, also known as record linkage, is a critical task in various domains of financial institutions like Société Générale. It involves the identification and linkage of references to the same real-world entities across different data sources. Effective entity matching is crucial for detecting suspicious transactions, ensuring compliance with national and international regulations. In this paper, we propose an approach that utilizes a bi-directional Gated Recurrent Unit (bi-GRU) model in the field of Natural Language Processing (NLP) to automate the process of automatic entity matching for Société Générale. Our results demonstrate that our bi-GRU model performs well in terms of accuracy, precision, recall, and AUC. The improved efficiency offered by our approach can significantly enhance Société Générale’s ability to detect potential money laundering activities and ensure regulatory compliance. More specifically, our approach brings a business value to Société Générale as it reduces potential costs: operational and regulatory costs.

# Methodology
**1. Dataset Construction**

Due to Société Générale’s activities, we did not have access to their data. They are highly confidential. Therefore we built our own dataset using a publicly available dataset. The original dataset consisted of a list of names (first name + last names).


*1.1. Synonym finder using Wikidata API*

We used Wikidata to search for alternative spellings of names. Wikidata is a collaboratively edited multilingual knowledge graph hosted by the Wikimedia Foundation. It is a common source of open data that Wikimedia projects such as Wikipedia, and anyone else, can use under the Creative Common public domain license. For each attribute or name, Wikidata builds a knowledge graph. 
For each name (first or last names), the API was looking for synonyms in Wikidata. 
Each name and attribute have their own ID number on Wikidata, for example, for the name Bill it is Q18245781 and the attribute ‘male given name’ is Q12308941. In order to get the equivalent names, we looked for the property ‘said to be the same as’ (instance with ID P460)
For example, the request ‘males given names known to be the same as Bill’ will be : element Q18245781 has a property P460 with value Q12308941.

The output of the request was the following:
Bill: Guillaume, William, Willem, Wilhelm, Guillén, Guilherme, Vilém, Vilmos, Guillermo, Gulielmus, Will, Gwilherm, Ghilherme, Gwilym, Viljem, Wiliam, Guglielmo.

We kept only synonyms with latin characters. However the scope should be extended to other alphabets.

Therefore, we built, for each name, a dictionary that has the initial name as key and all its synonyms as values. 
The example for Bill is as follows:
*{'Bill': [‘Guillaume’, ‘William’, ‘Willem’, ‘Wilhelm’, ‘Guillén’, ‘Guilherme’, ‘Vilém’, ‘Vilmos’, ‘Guillermo’, ‘Gulielmus’, ‘Will’, ‘Gwilherm’, ‘Ghilherme’, ‘Gwilym’, ‘Viljem’, ‘Wiliam’, ‘Guglielmo’]}*

In our case, the API will go through name items that respect the following condition: instances of Q12308941 and having a link of type P460 (‘said to be the same’).
This will be done to names that have not been retrieved so far, i.e. not contained in the dictionary created.


*1.2. Creation of positive and negative matches*

For each name, we created positive and negative matches. 
For the positive matches, we combined a synonym of the initial name, we did this for the initial first name and/or the initial last name. 

<img width="346" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/7acf086e-2bf9-4ffc-b25e-9e7cc895da01">

*creation of positive matches*

For the negative matches, we combined a first name and a last name that are not the same as the initial name, i.e. names not contained in the name dictionary. We simply split the name into words and search for other names containing at least one of the words from the original name. Full matches are excluded.

<img width="324" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/6229f870-280e-4b3b-9204-accf756868ef">

*creation of negative matches*

Therefore, once these two steps are completed, we obtained the following dataset:

*Sample of the dataset*

<img width="197" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/6eff419a-b94d-40fb-aa17-a08ba26c71d1">


**2. Model implementation and description**

In order to obtain the best performance we implemented different model architectures. The model that showed the best performance is a bidirectional Gated Recurrent Unit (Bi-GRU) - based deep Siamese network model.
A Siamese model is a type of neural network architecture that is designed to compare and measure similarity between two inputs - here, names. The architecture of a Siamese model (figure 5) consists of two identical subnetworks which share the same weights. Each subnetwork takes one name and processes it independently, generating a fixed-size embedding vector (the embedding size was 64 in our case). These embeddings capture the essential characteristics of the input name. 
We then compared the outputs from both networks using a similarity metric, a cosine similarity. The final prediction will be a linear function of this similarity. The closer to 1, the more similar the data. 

<img width="183" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/e27637f2-dc28-4ae0-a0de-ef01b1d24ff7">

*Architecture of our global siamese model*

Therefore, we defined two models:
- two identical subnetwork models
- a ‘global’ model: it incorporates the two subnetworks and the cosine similarity calculation.


*2.1. Description of our subnetworks*

The subnetworks play a crucial role in entity matching as they capture the characteristics of each name. They were the central part of our work. 
Different architectures were tested, such as two RNN-layers with or without normalization, a transformer-based model, etc. 

The architecture with the best performance is shown in figure 6 and consisted of: 
- *an embedding layer*: embeds each character into a representation vector.
- *a bidirectional Gated Recurrent Unit (GRU) layer*: processes sequential data in both forward and backward directions. It combines the information from past and future time steps to generate a comprehensive representation.  This layer can therefore understand the full context of the input. 
- *an attention layer*: applies attention mechanism by making a weighted sum of character representations.
- *a drop_out layer*: prevents overfitting
- *a dense layer*: performs a linear transformation on the attention output layer

<img width="522" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/337392a5-fc6d-4785-b5e8-0da2a71ef34b">

*Details of the subnetwork architecture*

This proposed architecture performed really well on  our training and validation datasets. The model training processus was the essential to achieve such a high performance

# Evaluation
In order to evaluate our model we used different metrics. We took into consideration:
-* Accuracy*: To prevent overfitting, we had to make sure that the gap between the accuracies on both training and validation dataset was not too big.
- *Binary cross-entropy*: we are looking to minimize the loss.
- *Recall*: It measures the proportion of relevant entities that are correctly identified by the matching model. It calculates the ratio of true positives to the sum of true positives and false negatives. 
- *Precision*: It measures the accuracy of the model in identifying positive matches. A higher precision indicates fewer false positives.
- *Area Under the Curve (AUC)*: overall performance of the model in terms of its ability to rank true matches higher than non-matches
In our project, to align our model with Société Générale’s conservative position, we mainly took into consideration the accuracy and the precision. Our aim was to reduce the false positive while not missing any more true negative (i.e. not creating false negatives). Therefore, it was a trade-off between false positives and false negatives: we were aiming for the lowest false positive rate that does not create false negatives.
AUC was also important in our process of evaluating the performance of our model.  In entity matching, the model needs to correctly prioritize and rank potential matches to optimize the matching process. Our AUC indicates that the model can effectively discriminate between matches and non-matches, facilitating efficient and accurate entity matching.
We achieved the following performance:

<img width="229" alt="image" src="https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/fe8d4039-446a-4ab0-ae3d-bebe8b5996a6">

Furthermore, we generated a confusion matrix to better visualize the false positives and false negatives rate.

![téléchargement (1)](https://github.com/Alinehbg/Automatic-Entity-Matching/assets/116564531/20d4eef4-5aa9-40f3-a2d2-060890a7f9db)

*Confusion matrix on the validation data*

We can visualize the false positive rate is 1.3% and the false negative rate is 1.9%. Those rates are the best we generated from our different architectures. 
Overall, we are satisfied with the final performance as it was obtained after a tedious process to test different architectures alongside different parameters. If time had allowed, we could have worked on improving further the performance.



