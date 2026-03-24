import os
import sys
import subprocess
import webbrowser
import time
import json
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_crop.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "CONFIG.json")
WEBAPP_PATH = os.path.join(BASE_DIR, "webapp.py")


def train_model():
    print("Training KNN model...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    data = pd.read_csv(config["raw_data_path"])
    x, y = data.drop("label", axis=1), data["label"]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.10, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(x_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    print("Model trained and saved.")


if not os.path.exists(MODEL_PATH):
    train_model()
else:
    print("Model already exists, skipping training.")

print("Launching CropSense website...")
time.sleep(1)
webbrowser.open("http://localhost:8501")
subprocess.run([sys.executable, "-m", "streamlit", "run", WEBAPP_PATH,
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false"])
