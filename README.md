# Analysing Affective Bias in Textual Emotion Detection System that utilize Large Pre-trained Language Model

This study aims to detect the existence of Affective biases in large Pre-trained Language Models (PLM), if any, concerning gender, and race, in context of textual emotion detection, distinctly from the widely studied general social biases. Here, the term Affective bias, is used to indicate any unfair association of emotions towards a particular gender, or race. That is, the system checks for any affective biases for example, whether representations of women are mostly being associated with a certain category of emotions like anger than men, or representations of any particular race are always being associated with a specific emotion, etc. Initially a textual emotion classification system is built using large PLMs, XLNet [1] and BERT [2] that is fine-tuned on SemEval 2018 Task 1: Affect in Tweets corpus [3] containing 8566 and 1464 data instances for training and validation respectively with four class of emotions ‘Anger’, ‘Fear’, ‘Joy’, and ‘Sadness’. To evaluate affective bias, performance of each model is analysed over 8400 sentences from one of the available evaluation corpus, Equity Evaluation Corpus (EEC) [4] that contains simple and small sentences synthetically generated using sentence templates for representing sentence pairs that only differ in gendered word or race word.


[download fine-tuned emotion prediction models](https://drive.google.com/drive/folders/1M_-B8WpByftRlk44tqQ9b-WrXv3rIj_X?usp=sharing) 
