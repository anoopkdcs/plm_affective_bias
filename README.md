# Analysing Affective Bias in Textual Emotion Detection System that utilize Large Pre-trained Language Model

<b> Methedology:</b> This study aims to detect the existence of Affective biases in large Pre-trained Language Models (PLM), if any, concerning gender, and race, in context of textual emotion detection, distinctly from the widely studied general social biases. Here, the term Affective bias, is used to indicate any unfair association of emotions towards a particular gender, or race. That is, the system checks for any affective biases for example, whether representations of women are mostly being associated with a certain category of emotions like anger than men, or representations of any particular race are always being associated with a specific emotion, etc. Initially a textual emotion classification system is built using large PLMs, XLNet [1] and BERT [2] that is fine-tuned on SemEval 2018 Task 1: Affect in Tweets corpus [3] containing 8566 and 1464 data instances for training and validation respectively with four class of emotions ‘Anger’, ‘Fear’, ‘Joy’, and ‘Sadness’. To evaluate affective bias, performance of each model is analysed over 8400 sentences from one of the available evaluation corpus, Equity Evaluation Corpus (EEC) [4] that contains simple and small sentences synthetically generated using sentence templates for representing sentence pairs that only differ in gendered word or race word.<br>
Fine-tuned emotion prediction models: :link: [download](https://drive.google.com/drive/folders/1M_-B8WpByftRlk44tqQ9b-WrXv3rIj_X?usp=sharing) 
<center><img src = 'https://github.com/anoopkdcs/plm_affective_bias/blob/main/plots/race_bert/1_anger_fulldata.png'></center>

## References <br>
[1] Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R.R., Le, Q.V.: Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems 32 (2019). <br>
[2] Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 4171–4186. Association for Computational Linguistics, Minneapolis, Minnesota (Jun 2019). <br>
[3] Kiritchenko, Svetlana, and Saif Mohammad. "Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems." Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics. 2018. <br>
[4] Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S. (2018, June). Semeval-2018 task 1: Affect in tweets. In Proceedings of the 12th International workshop on semantic evaluation (pp. 1-17).

