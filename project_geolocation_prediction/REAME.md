# DBL-Geolocation
Advanced System for Twitter-based Geolocation Prediction

### Introduction
This is a text-based geolocation prediction task. 

The *input* is plain text data and the *output* is one of the metro cities across the world.
Essentially, this is a multi-class classification problem.

The project uses *location indicative words (LIWs)* in text data to infer locations.
These LIWS include location names (*Sydney, London*), local dialectal words (*arvo for afternoon in Australia*) and topics (*Footy in Australia, Hockey in Canada and certain parts of USA*), etc.

One example is :
*I am taking a tram to Chinatown this arvo.* 
It doesn't mention any specific locations. Some indicative words are not unique to a city. 
For instance many cities have *Chinatown* and *tram* public transport. However, when combining dialectal words *arvo*, 
it is highly like the text is from Melbourne, Australia.

My approach to this problem is to use these LIWs in text for modelling and inference.
I've tried several rule-of-the-thumb algorithms, including Logistic Regressions, Multinomial Bayes, Multilayer Preceptron, Random Forest and Linear Support Vector Machines.
Additionally, I also tried a deep learning model, which uses a pre-trained DistilBert model, a dense hidden layer and an output layer architecture.
This is a typical transfer learning to fine-tune existing models for this particular task using the training data (i.e. standing on the shoulder of giants).

### Replication: Data, Code, Model and Results
#### Data
The original dataset requires download from Twitter and it takes time, thus I uploaded my download into S3 for replication)

Downloaded data (For data preprocessing, exploration and visualisation)

* sagemaker_train.jl:/dataset/sagemaker_train.jl
* sagemaker_validation.jl:/dataset/sagemaker_validation.jl
* sagemaker_test.jl:/dataset/sagemaker_test.jl

Transformed data after the data exploration (for scikit-learn and PyTorch training and evaluation)

* sagemaker_train.json(/dataset/sagemaker_train.json)
* sagemaker_validation.json(/dataset/sagemaker_validation.json)
* sagemaker_test.json(/dataset/sagemaker_test.json)

#### Code
Each script is self-contained. I've added requirements.txt for dependency installation for virtualenv.

* step1_data_preprocessing_exploration_visualisation.py
* step21_sklearn_train.py
* step22_pytorch_train.py (note: this assume using GPUS, I trained it in my local server due to cost reasons)
* step21_sklearn_eval.py
* step22_pytorch_eval.py (note: this also requires the exact GPU (i.e. cuda:1) as Pytorch model contains specific tensor data)
* requirements.txt

#### Model
These models can be directly re-used in evaluation scripts to generate results.
All scikit-learn models are ending with .joblib

* LR.joblib(/models/LR.joblib): Logistic Regression with L1 regularisations
* SVM.joblib(/models/SVM.joblib): Linear kernel SVM
* MB.joblib(/models/MB.joblib): Multinomial Bayes
* RF.joblib(/models/RF.joblib): Random Forest
* MLP.joblib(/models/MLP.joblib): Multilayer Preceptron

Fine-tuning of DistilBert models:

* pytorch_distilbert_9.bin(/models/pytorch_distilbert_9.bin): Bert model
* vocab_distilbert_9.bin(/models/vocab_distilbert_9.bin): Bert vocabulary

Resource file (required for evaluations):

* city2coords.jl(/dataset/city2coords.jl): city names mapping to lat/long values for distance-based metric
* label_map.json(/dataset/label_map.json): label maps for Bert model (can be regenerated on the fly)
* datalist.json(/modeltest/datalist.json): data list for visualization
