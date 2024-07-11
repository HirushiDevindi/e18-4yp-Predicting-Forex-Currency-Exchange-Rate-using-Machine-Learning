---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Predicting Forex Currency Exchange Rate using Machine Learning

<p align="center">
  <img src="./images/image0.jpg">
  <br>
</p>

#### Team

- E/18/375, Vindula K.P.A., [e18375@eng.pdn.ac.lk](mailto:e18375@eng.pdn.ac.lk)
- E/18/330, Sewwandi H.R., [e18330@eng.pdn.ac.lk](mailto:e18330@eng.pdn.ac.lk)
- E/18/323, Seekkubadu H.D., [e18323@eng.pdn.ac.lk](mailto:e18323@eng.pdn.ac.lk)

#### Supervisors

- Dr. Suneth Namal Karunarathna, [namal@eng.pdn.ac.lk](mailto:namal@eng.pdn.ac.lk)


#### Table of content

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Related works](#related-works)
4. [Methodology](#methodology)
5. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)
8. [Publications](#publications)
9. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract

<p style="text-align: justify;">
This research explores the potential of machine learning techniques in predicting forex currency exchange rates by leveraging a comprehensive set of technical and fundamental economic indicators. Recognizing the complexity and volatility of the foreign exchange market, our study seeks to develop a robust predictive model by integrating advanced machine learning algorithms. We aim to navigate the complexity of the foreign exchange market by identifying the most influential features through meticulous feature selection and designing novel architectures, refining our model’s performance through parameter optimization. The study anticipates contributing to a more accurate understanding of the factors driving forex market dynamics, yielding a reliable predictive model with potential applications for traders, investors, corporations, and policymakers.
</p>

## Introduction 

#### Forex Market

<p align="center">
  <img src="./images/image2.png">
  <br>
  <em>Figure 1.1:  OHLC prices in Forex market </em>
</p>

<p align="center">
  <img src="./images/image1.png">
  <br>
  <em>Figure 1.2:  Uptrend and Downtrend in Forex market </em>
</p>



<p style="text-align: justify;">
The forex market, also known as the foreign exchange market, is a global platform where participants engage in the buying and selling of currencies. It stands as the largest and most liquid financial market globally, with its daily trading volume exceeding $6 trillion. What distinguishes forex trading is its accessibility and constant activity, operating 24 hours a day, five days a week across different time zones. One of the key features of the forex market is its high liquidity, which means that traders can easily buy and sell currencies without significantly affecting their prices. This liquidity is driven by the participation of various entities, including central banks, financial institutions, corporations, governments, and individual traders. The forex market offers a wide range of currency pairs for trading, including major pairs like EUR/USD, GBP/USD, and USD/JPY, as well as minor and exotic pairs. These pairs create diverse market structures based on the combinations of currencies involved. For instance, while major pairs involve the world's strongest currencies, minor and exotic pairs may involve currencies from emerging or smaller economies.
</p>
<p style="text-align: justify;">
Traders analyze the forex market using various techniques, including fundamental analysis, technical analysis, and sentiment analysis. Fundamental analysis involves evaluating economic indicators, central bank policies, geopolitical events, and other factors that influence currency values. Technical analysis, on the other hand, relies on chart patterns, trend lines, and technical indicators to forecast price movements. Sentiment analysis gauges market sentiment and investor mood through factors such as news sentiment, social media activity, and trader positioning. OHLC (Open, High, Low, Close) prices are widely used in forex analysis, providing comprehensive information about price movements within specific time frames. These price data help traders identify trends, reversals, support and resistance levels, and other key trading opportunities.
</p>
<p style="text-align: justify;">
In recent years, advancements in technology, particularly in the field of machine learning, have enabled traders to develop sophisticated trading algorithms and models. These algorithms analyze vast amounts of historical and real-time data to identify patterns and trends, automate trading decisions, and manage risks more effectively. Overall, the forex market's dynamic nature, high liquidity, and accessibility make it a popular choice for traders seeking opportunities to profit from fluctuations in currency values around the world.
</p>

## Related works

<p style="text-align: justify;">
In our exploration of Forex currency exchange rate prediction, we have delved into various methodologies employed by scientists and experts across different domains. Recognizing the significance of predicting currency exchange rates, we've examined key models utilized in this endeavor. Our investigation has focused on bordering each area to uncover insights and refine our approach.
</p>
<p style="text-align: justify;">
Key Models Explored:
</p>

- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)

####  Convolutional Neural Networks (CNN)

<p align="center">
  <img src="./images/cnn.png">
  <br>
  <em>Figure 2.1:  CNN Architecture </em>
</p>

<p style="text-align: justify;">
A convolutional neural network (CNN) is a type of deep learning neural network that is specifically designed for image recognition and classification tasks. CNNs are inspired by the structure of the human visual cortex, and they are highly effective in a wide range of applications, including image classification, object detection, and facial recognition. The key characteristic of a CNN is its use of convolutional layers, which are made up of small filters (or kernels) that are applied to the input image. These filters extract features from the image, such as edges, lines, and corners. The output of the convolutional layer is then passed through a pooling layer, which reduces the dimensionality of the data and helps to make the network more robust to small variations in the input image. After the convolutional and pooling layers, the CNN typically has one or more fully connected layers, which are similar to the layers found in traditional neural networks. The fully connected layers classify the input image into one or more categories.
</p>
<p style="text-align: justify;">
For example, one of the research carried out under CNN to make the prediction is mentioned here. In the pursuit of accurate financial time series forecasting, researchers Alexiei Dingli and Karl Sant Fournier xie propose a novel deep learning approach leveraging Convolutional Neural Networks (CNNs). Their study, titled "Financial Time Series Forecasting – A Deep Learning Approach," introduces a CNN-based model developed using TensorFlow, fine-tuned for optimal performance. The model incorporates two convolutional and pooling layers, employing the same padding with a stride of 1 and ReLU activation functions for feature extraction. By harnessing the TensorFlow library's Argmax function, the researchers calculate accuracy while fine-tuning network parameters such as depth, learning rate, receptive area size, and feature count for each convolutional layer. Notably, their model achieves a commendable accuracy of 65% in forecasting the next month's price direction and 60% for predicting the next week's price direction, highlighting the effectiveness of deep learning methodologies in financial forecasting.
</p>

####  Recurrent Neural Networks (RNN)

<p align="center">
  <img src="./images/rnn.png">
  <br>
  <em>Figure 2.2:  RNN Architecture </em>
</p>

<p style="text-align: justify;">
RNNs are designed for sequential data processing. It includes feedback loops and feeds the output signal of a neuron back into the neuron. This way, information from previous time steps is preserved as the hidden state (ℎ terms). The objective of training a neural network is to minimize the value of a loss function, which represents the cumulative difference between the model’s outputs and the true labels. 
</p>

<p style="text-align: justify;">
In a study titled "Using Recurrent Neural Networks To Forecasting of Forex," researchers focused on training Recurrent Neural Networks (RNNs) to predict exchange rates between the American Dollar and four other major currencies: Japanese Yen, Swiss Franc, British Pound, and Euro. Employing the Elman-Jordan neural network method, they discovered that the most effective network configuration incorporated two crucial indicators: moving average and returns as inputs. To enhance accuracy, they replaced the moving average with the exponential moving average, which better approximates time series data and encapsulates information from longer periods. Their findings suggest that RNNs can predict the sign of increments in Forex rates with a high probability of approximately 80%, demonstrating practical viability for Forex forecasting applications.
</p>

<p style="text-align: justify;">
Recurrent Neural Networks (RNNs) often use activation functions such as the hyperbolic tangent (tanh) or the logistic sigmoid (σ). The derivative of both lies in the interval [0, 1]. Because of that two significant problems often arise during training,  gradient exploding and gradient vanishing. 
</p>

  - Gradient Exploding:
    <p style="text-align: justify;">
    Gradient exploding occurs when the gradients of the loss function become too large during backpropagation. This phenomenon leads to unstable training and makes it challenging to update the network parameters       effectively. As a result, the model's performance deteriorates, and the training process becomes unreliable.
    </p>
  - Gradient Vanishing:
    <p style="text-align: justify;">
    Gradient vanishing, on the other hand, happens when the gradients of the loss function become extremely small during backpropagation. In such cases, the gradients diminish as they propagate through time steps,     causing earlier time steps to have little to no influence on the parameter updates. Consequently, the RNN struggles to capture long-term dependencies in the data, limiting its ability to learn meaningful           patterns over extended sequences.
    </p>
    
<p style="text-align: justify;">
Both gradient exploding and gradient vanishing can hinder the training of RNNs and degrade their performance in tasks requiring the modeling of sequential data. These issues are particularly problematic in scenarios where the data exhibits long-term dependencies or temporal patterns that extend over many time steps.
</p>
<p style="text-align: justify;">
To address these challenges, various techniques have been proposed, including gradient clipping to prevent exploding gradients and architectural modifications such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) cells, which are designed to mitigate the vanishing gradient problem by allowing the network to selectively retain and update information over time.
</p>

####  Long Short-Term Memory (LSTM)

<p align="center">
  <img src="./images/lstm.png">
  <br>
  <em>Figure 2.3:  LSTM Architecture </em>
</p>

<p style="text-align: justify;">
Long Short-Term Memory (LSTM) networks are a powerful type of recurrent neural network (RNN) designed to overcome limitations in handling long-term dependencies present in traditional RNNs. LSTMs utilize "gates" to control the flow of information, allowing them to selectively remember and forget information relevant to the task at hand. This enables them to effectively process sequential data like speech, text, and time series data, making them valuable in applications like machine translation, speech recognition, and handwriting recognition.
</p>

<p style="text-align: justify;">
In the investigation titled "Using Long Short-Term Memory To Forecasting of Forex," researchers explored various LSTM approaches to predict Forex trends. Their study encompassed three primary methodologies: the Macroeconomic LSTM model, the Technical LSTM model, and the Hybrid LSTM Model. Among these, the Hybrid LSTM Model emerged as the most promising, showcasing superior predictive capabilities for Forex data. This comprehensive analysis sheds light on the effectiveness of LSTM-based techniques in forecasting Forex trends, with the hybrid approach demonstrating notable success in capturing and leveraging both macroeconomic and technical factors for enhanced prediction accuracy.
</p>


## Methodology

<p style="text-align: justify;">
We utilized historical data sourced from Yahoo! Finance. The dataset was split into three parts: 80% for training the model, 10% for validation, and 10% for testing. Feature selection focused on 11 technical indicators derived from the OHLC (Open, High, Low, Close) values of the currency pair. These indicators help in capturing market trends and price movements. We trained the model using two different combinations of technical indicators: one set with 9 indicators and another with all 11 indicators. This approach allowed us to compare the performance of the models and assess the impact of the number of indicators on predictive accuracy, ultimately helping to identify the most effective feature set for forecasting forex exchange rates.
</p>

###  Selected Technical Indicators

By considering the previous 5 days as a period, we are taking

- Close value of today
- Close value of yesterday
- Close value of the day before yesterday
- Close value of two days before yesterday
- Close  value of three days before yesterday

The following are the additional technical indicators:

- The simple moving average for 5 days for the High values
- The simple moving average for 5 days for the Low values
- The exponential moving average for 5 days for the Close value
- RSI - for Close value for 14 days time period
- The Moving Average Convergence Divergence (MACD)
- Price Rate of Change Indicator (ROC)

###  Model Architecture

<p align="center">
  <img src="./images/model.png">
  <br>
  <em>Figure 3.1:  Model Architecture </em>
</p>

<p style="text-align: justify;">
We created a sequential model with an LSTM layer followed by three dense layers. Through our evaluation, we identified LSTMs as the best performing model compared to CNNs and RNNs. LSTMs are preferred for time series prediction due to their ability to maintain and utilize long-term dependencies and handle time-dependent data effectively. Adding too many dense layers can lead to overfitting, where the model learns the training data too well but fails to generalize to new, unseen data. By using three dense layers, the model strikes a balance between learning complex patterns and avoiding overfitting. Networks using ReLU activation tend to converge faster during training compared to those using other activation functions.
</p>
<p style="text-align: justify;">
To optimize the model, we fine-tuned several parameters, including the number of LSTM units, the learning rate, and the evaluation frequency, to find the best-performing configuration. For training, we used mean square error (MSE) as the loss function, which measures the average of the squares of the errors—that is, the difference between the predicted and actual values. Additionally, we employed mean absolute error (MAE) as another performance metric to assess the model's accuracy, which measures the average magnitude of errors in a set of predictions, without considering their direction. This dual approach allowed us to gain a comprehensive understanding of the model's performance and ensure it effectively predicts forex exchange rates.
</p>

###  Hyperparameter Tuning

<p style="text-align: justify;">
  The first method we used for hyperparameter tuning was GridSearchCV. This approach involves providing a range of values for ephocs, LSTM units, and learning rate, and then iterating through these combinations to identify the one that minimizes the loss (Mean Square Error). The range of values was determined based on our intuitions and prior knowledge about the model, allowing us to systematically explore different configurations.
</p>
<p style="text-align: justify;">
The second method we employed was Bayesian optimization, a sophisticated approach that leverages probabilistic models to efficiently explore the hyperparameter space. Unlike GridSearchCV, which performs an exhaustive search, Bayesian optimization uses past evaluation results to build a probabilistic model of the objective function. In our case, we used Bayesian optimization to minimize the product of the training loss and the validation loss. This strategy aimed to find the best set of parameters that could effectively capture the underlying patterns in the data without overfitting to the training set. By balancing the training and validation losses, we ensured that the model generalized well to new, unseen data, leading to more robust and reliable predictions.
</p>

## Experiment Setup and Implementation

<p style="text-align: justify;">
  For hyperparameter optimization, we utilized two methods: GridSearchCV and Bayesian optimization. GridSearchCV involved providing a range of values for parameters such as EPOCHS, LSTM units, and learning rate, and systematically iterating through these combinations to identify the optimal configuration that minimized the loss (Mean Square Error). Bayesian optimization, on the other hand, employed probabilistic models to efficiently explore the hyperparameter space. By focusing on minimizing the product of the training loss and validation loss, Bayesian optimization helped us identify a set of parameters that effectively captured data patterns without overfitting.
</p>
<p style="text-align: justify;">
  We trained the models using two different input parameter sets: one with 9 technical indicators and another with 11 technical indicators derived from OHLC values. This allowed us to compare the performance and effectiveness of the models based on different feature sets, providing a comprehensive understanding of the impact of the number of indicators on the model's predictive accuracy.
</p>

##  Results and Analysis

<p align="center">
  <img src="./images/Results1.png">
  <br>
  <em>Figure 5.1:  Results for hyperparameters obtained from Grid Search CV method </em>
</p>

<p align="center">
  <img src="./images/Results2.png">
  <br>
  <em>Figure 5.2:  Results from hyperparameters obtained from Bayesian method </em>
</p>

<p align="center">
  <img src="./images/EURUSD.png">
  <br>
  <em>Figure 5.3:  Performance of EUR/USD model </em>
</p>

<p align="center">
  <img src="./images/LKRUSD.png">
  <br>
  <em>Figure 5.4:  Performance of LKR/USD model </em>
</p>

<p align="center">
  <img src="./images/GBPUSD.png">
  <br>
  <em>Figure 5.3:  Performance of GBP/USD model </em>
</p>

## Conclusion

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

 1. [Semester 7 Slides](./files/Semester%207%20Slides.pdf)
 2. [Semester 7 Report](./files/Semester%207%20Report.pdf)
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e18-4yp-Predicting-Forex-Currency-Exchange-Rate-using-Machine-Learning)
- [Project Page](https://cepdnaclk.github.io/e18-4yp-Predicting-Forex-Currency-Exchange-Rate-using-Machine-Learning)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
