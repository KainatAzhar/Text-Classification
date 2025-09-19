# Text Classification with a Fine-Tuned BERT Model

### 🎯 Project Overview
This project showcases a modern approach to Natural Language Processing (NLP) by fine-tuning a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for a specific text classification task. Unlike traditional models that rely on hand-crafted features, BERT learns contextual relationships between words, leading to state-of-the-art performance in sentiment analysis.

---

### 💾 Dataset
The model is fine-tuned on the **IMDb movie review dataset**, which contains 50,000 movie reviews labeled as either positive or negative. This dataset is a standard benchmark for sentiment analysis and is loaded directly from the Hugging Face `datasets` library.

---

### 📈 Methodology
1.  **Data Loading & Tokenization:** The IMDb dataset is loaded and preprocessed using BERT's specialized tokenizer (`bert-base-uncased`). This converts raw text into a numerical format, including `input_ids` and `attention_masks`, that the model can understand.

2.  **Model Loading:** A pre-trained `BertForSequenceClassification` model is loaded from the Hugging Face Transformers library. This powerful base model already possesses a deep understanding of the English language.

3.  **Fine-tuning:** The pre-trained model is fine-tuned on the IMDb dataset using the Hugging Face `Trainer` API. This process updates the model's weights to adapt it specifically to the sentiment classification task. Best practices such as using the `AdamW` optimizer are employed.

4.  **Evaluation:** The model's performance is evaluated on a held-out test set, with accuracy being the primary metric. The final fine-tuned model is saved for inference.

---

### 📊 Results
The fine-tuned BERT model achieves a high classification accuracy (typically over 90%), significantly outperforming simpler machine learning models. The project successfully demonstrates proficiency in using large language models, a critical skill in modern NLP, and an understanding of the fine-tuning process.

---

### 💻 Technologies Used
- Python
- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- Pandas
- Scikit-learn
- Jupyter Notebook / Google Colab

---

### 🚀 How to Run

1.  Clone this repository to your local machine.
2.  Create a virtual environment and activate it.
3.  Install the required libraries by running `pip install -r requirements.txt`.
4.  Open and run the Jupyter Notebook `bert_sentiment_analysis.ipynb` located in the `notebooks/` directory.
