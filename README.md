# GNN Benchmarking Documentation

![GitHub License](https://img.shields.io/github/license/Sambonic/gnn-benchmarking)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

Comparing between the performance of different GNNs on the same dataset 
#### Last Updated: January 4th, 2025

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)

<a name="installation"></a>
## Installation

Make sure you have [python](https://www.python.org/downloads/) downloaded if you haven't already.
Follow these steps to set up the environment and run the application:

1. Clone the Repository:
   
```bash
git clone https://github.com/Sambonic/gnn-benchmarking
```

```bash
cd gnn-benchmarking
```

2. Create a Python Virtual Environment:
```bash
python -m venv env
```

3. Activate the Virtual Environment:
- On Windows:
  ```
  env\Scripts\activate
  ```

- On macOS and Linux:
  ```
  source env/bin/activate
  ```
4. Ensure Pip is Up-to-Date:
  ```
  python.exe -m pip install --upgrade pip
  ```
5. Install Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Import GNN Benchmarking as shown below.


<a name="usage"></a>
## Usage
## GNN Model Benchmarking Project Walkthrough

This project benchmarks four different Graph Neural Network (GNN) models on a Reddit dataset.  The steps below detail the process, assuming you've already cloned the repository and installed the necessary libraries.

**1. Data Loading and Preprocessing:**

The notebook begins by loading the `webis/tldr-17` dataset from Hugging Face.  This dataset contains Reddit posts, including their content, authors, subreddits, and more. The code then converts the Hugging Face dataset to a Pandas DataFrame for easier manipulation.

```python
# load the reddit dataset from HuggingFace repositories
ds = load_dataset(
    "webis/tldr-17",
    trust_remote_code=True,
    split="train[:100%]",)

# convert the HuggingFace dataset to pandas for easier use
df = ds.to_pandas()
```

Next, it filters the dataset to include only the top 3 subreddits and downsamples the data to balance the class distribution.  Text cleaning is performed using a custom function `clean_text`, which removes digits, non-word characters, single-character words, and stop words.

```python
# ... (subreddit filtering and downsampling code) ...

def clean_text(text):
  # ... (text cleaning logic) ...

df["content"] = df["content"].apply(clean_text)
```

**2. Text Embedding and Feature Encoding:**

Sentence embeddings are generated using the `all-MiniLM-L6-v2` SentenceTransformer model. These embeddings represent the semantic meaning of the post content.  Subreddits, authors, and post IDs are then label encoded to numerical representations.  Unnecessary columns are dropped from the DataFrame.

```python
# use Microsoft's all-MiniLM-L6-v2 model for text embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to('cuda') # move the model to GPU if available
embeddings = model.encode(df['content'].tolist(), device='cuda')
df['content'] = list(embeddings)

# ... (Label encoding for subreddits, authors, and post IDs) ...
```

**3. Graph Creation:**

A heterogeneous graph is constructed using `torch_geometric.data.HeteroData`.  The graph consists of:

* **Nodes:**  `post` nodes (with content embeddings as features and subreddit as labels), and `author` nodes (with placeholder features).
* **Edges:** `author`-to-`post` edges representing authorship, and `post`-to-`post` edges representing similarity between posts (based on cosine similarity of their embeddings). A threshold is used to determine which pairs of posts are considered similar.

```python
data = HeteroData()
# ... (Adding post nodes, author nodes, and edges) ...
```
A visualization of a smaller subgraph is created using `networkx` to illustrate the graph structure.


**4. Data Splitting and Masking:**

The dataset is split into training, validation, and testing sets using stratified sampling to maintain class proportions.  Boolean masks are created to easily select the relevant data for each set.

```python
# ... (train_test_split and mask creation code) ...
```


**5. Model Definition and Training:**

Four different GNN models are defined: `GraphSAGE`, `GatedGraphConv`, `GCN`, and `GIN`. These models use different architectures to process the graph data and predict the subreddit of a post.  The `train_model` function trains each model using the Adam optimizer and cross-entropy loss, logging training progress to TensorBoard.

```python
class GraphSAGE(torch.nn.Module):
    # ... (GraphSAGE model definition) ...

class GatedGNN(torch.nn.Module):
    # ... (GatedGNN model definition) ...

class GCN(torch.nn.Module):
    # ... (GCN model definition) ...

class GIN(torch.nn.Module):
    # ... (GIN model definition) ...

# ... (Training loop using train_model function) ...
```

**6. Model Evaluation:**

The `test_model` function evaluates each trained model on the test set, calculating accuracy, precision, recall, and F1-score.  The results are logged to TensorBoard.  The notebook also generates various plots to visualize the training progress and model performance comparison.  A grouped bar chart compares the evaluation metrics for all the models and another chart shows the training time and memory consumption of the models.  Finally, the confusion matrices are plotted for each model to visualize the classification performance.

```python
def test_model(model, X_test, y_test, edge_index, log_dir='./runs'):
    # ... (Testing and metric calculation code) ...


# ... (Plotting code) ...
```

**7. Hyperparameter Tuning:**

A section is dedicated to hyperparameter tuning where a dictionary `test_dict` specifies the hyperparameters to test and the `test_hyperparams` functions iterate through all combinations using the updated `train_model` function that saves results per hyperparameter combination to TensorBoard.


**To run this project:**

1. **Ensure you have a compatible environment set up:**  This project requires Python with the libraries specified in the `%pip install` line at the beginning. A GPU is recommended but not strictly required.
2. **Clone the repository and install the required libraries.** (Instructions not included here as requested)
3. **Run the Jupyter Notebook:** Execute each code cell sequentially.  The TensorBoard visualizations will be available after training is complete.  Open TensorBoard by running `%tensorboard --logdir runs` in a Jupyter Notebook cell (or from your command line: `tensorboard --logdir runs`).


This detailed walkthrough provides a comprehensive understanding of the project's functionality and execution.  Remember that the training process may take a considerable amount of time depending on your hardware.


<a name="features"></a>
## Features
- **Reddit Post Classification:**  Classifies Reddit posts into subreddits using graph neural networks (GNNs).
- **Heterogeneous Graph Construction:** Creates a heterogeneous graph representing posts, authors, and relationships between them (e.g., authorship, post similarity).
- **Text Embedding:** Uses SentenceTransformer ('all-MiniLM-L6-v2') to generate embeddings for post content.
- **GNN Model Benchmarking:** Trains and evaluates multiple GNN architectures (GraphSAGE, GatedGraphConv, GCN, GIN) on the constructed graph.
- **Hyperparameter Tuning:**  Experiments with different hyperparameters for each GNN model to find optimal settings.
- **Performance Evaluation:**  Assesses model performance using metrics like accuracy, precision, recall, and F1-score, along with visualization of learning curves and confusion matrices.
- **Resource Monitoring:** Tracks and reports training time and peak memory usage for each model.
- **TensorBoard Integration:** Uses TensorBoard for visualization and logging of training progress, metrics, and model graphs.


