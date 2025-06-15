# 🎬 Analyse de sentiment de critiques de films IMDb

Ce projet utilise **TensorFlow** pour entraîner un modèle qui prédit si une critique de film est **positive** ou **négative**.

---

## 🧠 Modèle

- 📚 Dataset : IMDb (25 000 critiques d'entraînement/test)
- 🧾 Prétraitement : padding des séquences à 256 mots
- 🧱 Architecture :
  - `Embedding`
  - `GlobalAveragePooling1D`
  - `Dense (ReLU)`
  - `Dense (Sigmoid)`

---

## ⚙️ Exemple

```python
sample_text = "I really loved this movie, it was amazing"
