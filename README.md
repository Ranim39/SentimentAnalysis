# IMDb Movie Review Sentiment Analysis 🎬

This project uses **TensorFlow** to train a model that predicts whether a movie review is **positive** or **negative**.


sample_text = "I really loved this movie, it was amazing"
![image](https://github.com/user-attachments/assets/697ad5e8-15a5-44ef-b72c-83cea383f14d)

---

## 📊 Dataset

- **Source**: IMDb (25,000 training reviews + 25,000 test reviews)
- **Data type**: Raw text (movie reviews)

---

## 🧠 Model

- **Preprocessing**:
  - Tokenization
  - Padding sequences to a fixed length (256 words)
- **Architecture**:
  - `Embedding` layer
  - `GlobalAveragePooling1D`
  - `Dense` (ReLU)
  - `Dense` (Sigmoid)

---

## ⚙️ Example

```python
sample_text = "I really loved this movie, it was amazing"
model.predict([sample_text])
# ➜ Returns a probability between 0 (negative) and 1 (positive)
