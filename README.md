## 🎬 Movie Recommendation System

This is a **content-based movie recommender system** built using the **TMDb 5000 Movie Dataset**. It uses **TF-IDF vectorization** on movie genres and descriptions to recommend similar movies. A simple **Streamlit web interface** is included for user interaction.

---

### 🚀 Features

* Recommends movies based on description and genres
* Clean, easy-to-use Streamlit interface
* Fast and responsive recommendations
* Lightweight – no user ratings required

---

### 📁 Dataset

**Source:** [TMDb 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Download and place the file `tmdb_5000_movies.csv` in your project directory.

---

### 🛠 Installation

1. **Clone the repository** or copy the code files.

2. **Install required Python packages**:

   ```bash
   pip install streamlit pandas scikit-learn
   ```

3. **Download the dataset**:

   * Go to [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
   * Download and extract the file
   * Place `tmdb_5000_movies.csv` in the same directory as `app.py`

---

### ▶️ Run the App

```bash
streamlit run app.py
```

Open the provided `localhost` URL in your browser.

---

### 🧠 How It Works

1. Loads and preprocesses movie data (`overview`, `genres`)
2. Converts text into numerical vectors using **TF-IDF**
3. Calculates **cosine similarity** between movies
4. Returns the top N most similar movies for a selected title

---

### 📦 Project Structure

```
├── app.py                # Streamlit app
├── tmdb_5000_movies.csv  # Movie metadata
├── README.md             # Project README
