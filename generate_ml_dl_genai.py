#!/usr/bin/env python3
"""
Script to generate ML, DL, and GenAI notebooks
"""
import json

def create_notebook(title, sections):
    cells = []
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {title}\n\n---\n']})
    
    for section in sections:
        if 'markdown' in section:
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': section['markdown']})
        if 'code' in section:
            cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': section['code']})
    
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'}
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

notebooks = {
    # MACHINE LEARNING
    '04-machine-learning/01-ml-introduction.ipynb': {
        'title': '🤖 Machine Learning: Introduction',
        'sections': [
            {'markdown': ['## 📚 What is Machine Learning?\n\nMachine Learning is a subset of AI that enables computers to learn from data.\n\n**Types:**\n- Supervised Learning\n- Unsupervised Learning\n- Reinforcement Learning\n']},
            {'code': ['# Installing scikit-learn\n# pip install scikit-learn\n\nimport sklearn\nprint(f"Scikit-learn version: {sklearn.__version__}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ ML learns from data\n✅ Supervised: labeled data\n✅ Unsupervised: unlabeled data\n']}
        ]
    },
    '04-machine-learning/02-numpy-pandas.ipynb': {
        'title': '🤖 ML: NumPy & Pandas',
        'sections': [
            {'markdown': ['## 📊 NumPy Arrays\n']},
            {'code': ['import numpy as np\n\n# Creating arrays\narr = np.array([1, 2, 3, 4, 5])\nprint(f"Array: {arr}")\nprint(f"Shape: {arr.shape}")\nprint(f"Mean: {arr.mean()}")\n\n# 2D array\nmatrix = np.array([[1, 2, 3], [4, 5, 6]])\nprint(f"Matrix:\\n{matrix}")']},
            {'markdown': ['## 📈 Pandas DataFrames\n']},
            {'code': ['import pandas as pd\n\n# Creating DataFrame\ndata = {\n    "Name": ["Alice", "Bob", "Charlie"],\n    "Age": [25, 30, 35],\n    "City": ["NYC", "LA", "Chicago"]\n}\ndf = pd.DataFrame(data)\nprint(df)\nprint(f"\\nDescribe:\\n{df.describe()}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ NumPy for numerical computing\n✅ Pandas for data manipulation\n✅ DataFrames are powerful\n']}
        ]
    },
    '04-machine-learning/03-data-preprocessing.ipynb': {
        'title': '🤖 ML: Data Preprocessing',
        'sections': [
            {'markdown': ['## 🧹 Handling Missing Data\n']},
            {'code': ['import pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({\n    "A": [1, 2, np.nan, 4],\n    "B": [5, np.nan, 7, 8]\n})\nprint("Original:\\n", df)\nprint("\\nFilled:\\n", df.fillna(0))']},
            {'markdown': ['## ⚖️ Feature Scaling\n']},
            {'code': ['from sklearn.preprocessing import StandardScaler\n\ndata = [[1], [2], [3], [4], [5]]\nscaler = StandardScaler()\nscaled = scaler.fit_transform(data)\nprint(f"Scaled:\\n{scaled}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Clean data before training\n✅ Handle missing values\n✅ Scale features appropriately\n']}
        ]
    },
    '04-machine-learning/04-supervised-learning.ipynb': {
        'title': '🤖 ML: Supervised Learning',
        'sections': [
            {'markdown': ['## 📈 Linear Regression\n']},
            {'code': ['from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\n# Sample data\nX = np.array([[1], [2], [3], [4], [5]])\ny = np.array([2, 4, 6, 8, 10])\n\n# Train model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Predict\nprediction = model.predict([[6]])\nprint(f"Prediction for 6: {prediction[0]}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Linear Regression for continuous targets\n✅ Logistic Regression for classification\n✅ Train-test split prevents overfitting\n']}
        ]
    },
    '04-machine-learning/05-unsupervised-learning.ipynb': {
        'title': '🤖 ML: Unsupervised Learning',
        'sections': [
            {'markdown': ['## 🎯 K-Means Clustering\n']},
            {'code': ['from sklearn.cluster import KMeans\nimport numpy as np\n\n# Sample data\nX = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])\n\n# Clustering\nkmeans = KMeans(n_clusters=2, random_state=0)\nkmeans.fit(X)\nprint(f"Labels: {kmeans.labels_}")\nprint(f"Centers:\\n{kmeans.cluster_centers_}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ K-Means groups similar data\n✅ PCA reduces dimensions\n✅ No labels needed\n']}
        ]
    },
    '04-machine-learning/06-model-evaluation.ipynb': {
        'title': '🤖 ML: Model Evaluation',
        'sections': [
            {'markdown': ['## 📊 Evaluation Metrics\n']},
            {'code': ['from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\nimport numpy as np\n\ny_true = [0, 1, 1, 0, 1]\ny_pred = [0, 1, 0, 0, 1]\n\nprint(f"Accuracy: {accuracy_score(y_true, y_pred)}")\nprint(f"Precision: {precision_score(y_true, y_pred)}")\nprint(f"Recall: {recall_score(y_true, y_pred)}")\nprint(f"F1 Score: {f1_score(y_true, y_pred)}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Multiple metrics needed\n✅ Cross-validation prevents overfitting\n✅ Grid search tunes hyperparameters\n']}
        ]
    },
    
    # DEEP LEARNING
    '05-deep-learning/01-neural-networks-basics.ipynb': {
        'title': '🧠 Deep Learning: Neural Networks',
        'sections': [
            {'markdown': ['## 🔬 What are Neural Networks?\n\nNeural networks are computing systems inspired by biological neural networks.\n\n**Components:**\n- Neurons (nodes)\n- Layers (input, hidden, output)\n- Weights and biases\n- Activation functions\n']},
            {'code': ['# Simple perceptron\nimport numpy as np\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n\n# Forward pass\ninputs = np.array([1.0, 2.0, 3.0])\nweights = np.array([0.5, -0.5, 0.2])\nbias = 0.1\n\noutput = sigmoid(np.dot(inputs, weights) + bias)\nprint(f"Output: {output}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ NNs learn hierarchical features\n✅ Backpropagation updates weights\n✅ Activation functions introduce non-linearity\n']}
        ]
    },
    '05-deep-learning/02-tensorflow-keras.ipynb': {
        'title': '🧠 DL: TensorFlow & Keras',
        'sections': [
            {'markdown': ['## 🔥 Building Models with Keras\n']},
            {'code': ['# pip install tensorflow\nimport tensorflow as tf\nfrom tensorflow import keras\n\n# Simple model\nmodel = keras.Sequential([\n    keras.layers.Dense(64, activation="relu", input_shape=(10,)),\n    keras.layers.Dense(32, activation="relu"),\n    keras.layers.Dense(1, activation="sigmoid")\n])\n\nmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])\nprint(model.summary())']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Keras high-level API\n✅ Sequential for linear stacks\n✅ Compile before training\n']}
        ]
    },
    '05-deep-learning/03-cnns.ipynb': {
        'title': '🧠 DL: CNNs',
        'sections': [
            {'markdown': ['## 🖼️ Convolutional Neural Networks\n\nCNNs excel at image processing.\n\n**Key Layers:**\n- Conv2D: Extract features\n- MaxPooling: Reduce dimensions\n- Flatten: Prepare for dense layers\n']},
            {'code': ['import tensorflow as tf\nfrom tensorflow import keras\n\nmodel = keras.Sequential([\n    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),\n    keras.layers.MaxPooling2D((2, 2)),\n    keras.layers.Flatten(),\n    keras.layers.Dense(10, activation="softmax")\n])\n\nprint(model.summary())']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ CNNs for images\n✅ Convolution extracts features\n✅ Pooling reduces size\n']}
        ]
    },
    '05-deep-learning/04-rnns-lstm.ipynb': {
        'title': '🧠 DL: RNNs & LSTM',
        'sections': [
            {'markdown': ['## 🔄 Recurrent Neural Networks\n\nRNNs process sequential data.\n\n**Applications:**\n- Time series\n- Text generation\n- Speech recognition\n']},
            {'code': ['import tensorflow as tf\nfrom tensorflow import keras\n\nmodel = keras.Sequential([\n    keras.layers.LSTM(64, input_shape=(10, 1)),\n    keras.layers.Dense(1)\n])\n\nprint(model.summary())']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ RNNs for sequences\n✅ LSTM handles long-term dependencies\n✅ GRU is simpler alternative\n']}
        ]
    },
    '05-deep-learning/05-transfer-learning.ipynb': {
        'title': '🧠 DL: Transfer Learning',
        'sections': [
            {'markdown': ['## 🔄 Transfer Learning\n\nUse pre-trained models as starting point.\n\n**Benefits:**\n- Less training data needed\n- Faster training\n- Better performance\n']},
            {'code': ['import tensorflow as tf\nfrom tensorflow.keras.applications import VGG16\n\n# Load pre-trained model\nbase_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))\nbase_model.trainable = False\n\nprint(f"Loaded {base_model.name}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Leverage existing knowledge\n✅ Freeze base layers\n✅ Fine-tune for your task\n']}
        ]
    },
}

# Create notebooks
for filepath, config in notebooks.items():
    nb = create_notebook(config['title'], config['sections'])
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'✅ Created {filepath}')

print('\n🎉 Generated ML and DL notebooks!')
print('📝 Next: Creating GenAI notebooks with RAG, Agents, and MCP...')
