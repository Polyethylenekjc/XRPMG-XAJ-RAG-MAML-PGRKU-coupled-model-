**XRPMG Project Documentation**

-----

## **Project Overview**

**XRPMG (XAJ-RAG-MAML-PGRKU coupled model)** is a general and highly integrated learning framework focused on hydrological time series analysis and prediction in complex watersheds.

The framework integrates data-driven learning with physical constraints via the **PGRKU model**, uses a **meta-learning (MAML) strategy** to enhance generalization across different watersheds, and incorporates a **Retrieval-Augmented Generation (RAG) mechanism** to dynamically integrate relevant historical information, thereby maintaining high prediction accuracy during extreme events (such as floods).

-----

## **Key Features**

  - **Hybrid Learning Architecture**: Combines data-driven deep learning with physical meta-learning to handle complex, non-linear hydrological processes.
  - **Meta-Learning Strategy**: Employs the MAML algorithm, enabling the model to quickly adapt to unseen watershed data, significantly improving its generalization capability.
  - **Physical Constraint Integration**: Embeds physical laws and hydrological process knowledge into the model to ensure prediction results align with hydraulic principles.
  - **Retrieval-Augmented Generation**: Dynamically retrieves information from historical data/knowledge bases to provide key context for the model.
  - **Feature Explanation Function**: Provides a SHAP-based explanation tool to analyze the importance of features behind the model's predictions.

-----

## **Technical Architecture**

### **Core Components**

1.  **PGRKU Prediction Model (`PGRKUPredictor`)**

      - Captures complex dependencies in time series.
      - Integrates physical constraints.
      - Optimized specifically for hydrological prediction tasks.

2.  **Meta-Learning Module (MAML Module)**

      - Implements model-agnostic meta-learning strategies.
      - Supports rapid adaptation to new tasks.

3.  **Retrieval-Augmented Module (RAG Module)**

      - Retrieves relevant information from historical data/knowledge bases.
      - Combines retrieval results with input to improve prediction accuracy.

4.  **Data Preprocessing Module**

      - Normalization.
      - Missing value imputation.
      - Feature engineering.

5.  **Analysis and Visualization Module**

      - Provides post-hoc explanation tools like SHAP.

-----

## **Project Structure**

```bash
├── Config/                   # Configuration files directory
├── src/                      # Source code directory
│   ├── analyzer/             # Analyzer module
│   ├── data/                 # Data processing module
│   ├── model/                # Model definitions
│   ├── trainer/              # Training logic
│   └── utils/                # Utility classes
├── XAJ/                      # XAJ model-related code
├── README.md                 # Project documentation
├── config_schema.json        # Configuration file schema
└── config_gui.py             # GUI configuration window
```

## **Installation and Dependencies**

Please use the following command to install the dependencies:

```bash
pip install torch numpy pandas scikit-learn shap faiss-cpu
```

## **Configuration Guide**

### **Configuration File Structure**

Configuration files are located in the `Config/` directory and use the **YAML** format.
`config_schema.json` is used to validate the configuration file format.

The main sections include:

  - **data**: Dataset configuration block

      - `datasets`: List of datasets (`name`, `processors`, `path`, `target_column`)
      - `processors`: Data preprocessing steps and parameters
      - `window`: Sliding window size
      - `test_size`: Test set ratio

  - **models**: Model configuration

      - Each model is defined by `name`.
      - Includes `enabled` and `type` fields.
      - Optional `params`: Model hyperparameters.
      - Optional `trainer`: Training configuration (e.g., cross-validation `cv`, evaluation metrics `scoring`).

  - **analyzers**: Post-experiment analysis tools configuration.

  - **output**: Experiment results output paths (`analysis_result_path`, `data_result_path`).

-----

## **Usage Instructions**

1.  **Prepare Data**
    Save the time series data as a **CSV file**, ensuring it contains all relevant feature columns and the target column.

2.  **Configure the Experiment**
    Modify the `Config/test.yaml` file, or use the provided GUI tool:

    ```bash
    python config_gui.py
    ```

3.  **Run the Experiment**
    Execute the main script:

    ```bash
    python -m src.run
    ```

## **Technical Innovations**

  - **Coupling of Physical Knowledge and Deep Learning**
    Incorporates physical constraints into the PGRKU model, making prediction results more interpretable and physically sound.

  - **Adaptive Generalization Capability**
    Utilizes the MAML meta-learning method, allowing the model to quickly adapt to new watersheds and reducing its reliance on large amounts of historical data.

  - **Context-Aware Prediction**
    Leverages the RAG module to dynamically retrieve relevant historical information, effectively addressing extreme events like floods.

-----

## **Performance Metrics**

The framework supports multi-dimensional evaluation metrics, including:

  - **Prediction Accuracy**: MAE, MSE, RMSE, R²
  - **Hydrological Metrics**: NSE, KGE
  - **Physical Constraint Satisfaction**: Constraint violation penalty score

-----

## **Future Enhancements**

  - **Multi-Source Data Integration**: Fuse external data sources such as weather forecasts and satellite remote sensing data.
  - **Uncertainty Quantification**: Provide methods for assessing the uncertainty of prediction results.
  - **Reinforcement Learning Optimization**: Explore advanced RL algorithms to improve long-term scheduling and decision-making capabilities.

-----

## **Contact Information**

Due to the confidentiality of engineering parameters and data security considerations, this document provides only the core algorithm framework.

For access to the complete source code, datasets, and technical documentation, please contact:
📧 **kjc2005105@126.com**