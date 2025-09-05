import os
from typing import Any, Dict
import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import gridspec
from scipy.spatial import KDTree
import seaborn as sns
from src.utils.Logger import Logger
from src.analyzer.analyzerBase import AnalyzerBase
from src.factory.analyzer_factory import AnalyzerFactory

@AnalyzerFactory.register('shap_analyzer')
class ShapAnalyzer(AnalyzerBase):
    """
    An analyzer using SHAP to provide interpretability for a specified model.
    It supports the following features:
    - SHAP value calculation
    - KMeans clustering
    - Adding true and predicted labels
    - Marking prediction accuracy (small error: 1, large error: 0)
    - Feature importance ranking based on SHAP values
    - Dynamic jitter plot visualization
    - Marking correctly/incorrectly predicted samples
    - Feature trajectory lines
    - Cluster group visualization
    - Cluster center heatmap
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config)
        # Prioritize fetching from kwargs, then fallback to config
        self.model_name = kwargs.get("model_name", (config or {}).get("model_name"))
        self.threshold = kwargs.get("error_threshold", (config or {}).get("error_threshold", 0.1)) # Error threshold, defaults to 10%

    def _save_circos_data(self, path: str) -> None:
        """
        Saves data for drawing a Circos plot.
        """
        if self.res is None:
            self.logger.warning("No results to save for Circos data")
            return

        shap_values_df = self.res["shap_values_df"]
        predictions = self.res.get("predictions", {})

        if not predictions:
            self.logger.warning("No model prediction information available, cannot save Circos data")
            return

        # Get the predicted values of the first model
        model_name = next(iter(predictions.keys()))
        pred_values = predictions[model_name]

        # First data: Length mapping for each cluster
        cluster_lengths = {}
        plot_length = len(shap_values_df)
        cluster_lengths["Plot"] = plot_length

        # Calculate the length of each cluster
        cluster_counts = shap_values_df["Cluster"].value_counts()
        for cluster in ['A', 'B', 'C', 'D']:
            cluster_lengths[cluster] = int(cluster_counts.get(cluster, 0))

        # Second data: Position mapping of each cluster in the plot
        cluster_positions = {}
        for cluster in ['A', 'B', 'C', 'D']:
            # Find all indices for this cluster
            cluster_indices = shap_values_df[shap_values_df["Cluster"] == cluster].index.tolist()
            if not cluster_indices:
                cluster_positions[cluster] = []
                continue

            # Group continuous indices into intervals
            intervals = []
            start = cluster_indices[0]
            end = cluster_indices[0]

            for i in range(1, len(cluster_indices)):
                if cluster_indices[i] == end + 1:
                    end = cluster_indices[i]
                else:
                    intervals.append([start, end + 1])
                    start = cluster_indices[i]
                    end = cluster_indices[i]
            intervals.append([start, end + 1])

            # Construct the mapping, ensuring internal and external intervals have the same length
            cluster_positions[cluster] = []
            for interval in intervals:
                interval_length = interval[1] - interval[0]
                # Internal interval starts from 0 with the same length as the external one
                internal_start = 0 if not cluster_positions[cluster] else cluster_positions[cluster][-1][0][1]
                internal_end = internal_start + interval_length
                cluster_positions[cluster].append([
                    [internal_start, internal_end], # Relative position within the cluster
                    [interval[0], interval[1]]      # Absolute position within the Plot
                ])

        # Third data: Cluster distribution for the Plot segment
        plot_segments = []
        cluster_colors = {
            'A': '#E74C3C',  # Coral red
            'B': '#2ECC71',  # Emerald green
            'C': '#3498DB',  # Royal blue
            'D': '#F39C12',  # Orange
            'Plot': '#95A5A6' # Stone gray
        }

        # Group continuous intervals by cluster
        current_cluster = shap_values_df.iloc[0]["Cluster"]
        start_idx = 0

        for i in range(1, len(shap_values_df)):
            if shap_values_df.iloc[i]["Cluster"] != current_cluster:
                plot_segments.append([
                    start_idx, 
                    i, 
                    cluster_colors[current_cluster]
                ])
                current_cluster = shap_values_df.iloc[i]["Cluster"]
                start_idx = i

        # Add the last interval
        plot_segments.append([
            start_idx, 
            len(shap_values_df), 
            cluster_colors[current_cluster]
        ])

        # Fourth data: SHAP importance for each cluster (log-transformed)
        cluster_shap_importance = {}

        # Get SHAP values for the top 13 features
        feature_names = self.res["feature_names"][:13]
        shap_values_filtered = shap_values_df[feature_names]

        for cluster in ['A', 'B', 'C', 'D']:
            cluster_data = shap_values_df[shap_values_df["Cluster"] == cluster]
            cluster_length = len(cluster_data)

            if cluster_length == 0:
                cluster_shap_importance[cluster] = [[], [], []]
                continue

            # Calculate the average SHAP value for the cluster
            cluster_shap_values = shap_values_filtered.loc[cluster_data.index]
            mean_shap = cluster_shap_values.mean()

            # Log-transform, preserving the original sign
            # Use sign(x) * log(|x| + 1) to handle log(0) and maintain the sign
            log_shap = np.sign(mean_shap) * np.log(np.abs(mean_shap) + 1e-10)

            # Calculate x-coordinates (symmetrically centered selection)
            num_features = len(feature_names)

            if cluster_length <= num_features:
                # If cluster length is less than or equal to the number of features, use all points, but ensure coordinates are within range
                # Use a uniform distribution to make sure it doesn't exceed cluster_length
                x_coords = [int(i * (cluster_length - 1) / (num_features - 1)) 
                             if num_features > 1 else 0 
                             for i in range(min(num_features, cluster_length))]
            else:           
                start_idx = int(cluster_length*0.1)
                end_idx = int(cluster_length*0.9)

                # Distribute coordinates uniformly within the valid range
                num_selected = end_idx - start_idx
                if num_selected > 1:
                    x_coords = [start_idx + int(i * (num_selected - 1) / (num_features - 1)) 
                                 for i in range(min(num_features, num_selected))]
                else:
                    x_coords = [start_idx] if num_features > 0 else []

            cluster_shap_importance[cluster] = [x_coords, log_shap.tolist(), feature_names]
        # Save data to file
        circos_data = {
            "cluster_lengths": cluster_lengths,
            "cluster_positions": cluster_positions,
            "plot_segments": plot_segments,
            "cluster_shap_importance": cluster_shap_importance
        }

        import json
        with open(os.path.join(path, "circos_data.json"), "w") as f:
            json.dump(circos_data, f, indent=2)

    def _analyze(self, predictions: Dict[str, Any], true_values: Any, model=None, X_test=None, feature_names=None, **kwargs) -> None:
        """
        Performs SHAP analysis on the specified model and adds clustering, labels, and prediction accuracy analysis.
        """
        if not model or X_test is None or feature_names is None:
            raise ValueError("SHAP analysis requires a model, test data, and feature names")

        device = next(model.parameters()).device
        self.logger.info(f"Running SHAP analysis, model name: {self.model_name}", module=self.__class__.__name__)
        self.logger.debug(f"Using model: {model}", module=self.__class__.__name__)
        self.logger.debug(f"X_shape: {X_test.shape}, y_shape: {true_values.shape}", module=self.__class__.__name__)

        X_test_np = None

        try:
            model.eval()
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_test_tensor = torch.from_numpy(X_test_np).float().to(device).detach()
        except Exception as e:
            self.logger.warning(f"Failed to convert data to Tensor, details: {e}", module=self.__class__.__name__)
            X_test_tensor = X_test

        if hasattr(model, 'is_rag_frozen') and not model.is_rag_frozen():
            model.freeze_rag()

        try:
            explainer = shap.GradientExplainer(model, X_test_tensor)
            model.train()
            shap_values = explainer.shap_values(X_test_tensor)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to run SHAP analysis: {e}")

        shap_values_avg = shap_values.mean(axis=1).squeeze()
        shap_values_df = pd.DataFrame(shap_values_avg, columns=feature_names)

        # Add cluster information
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(shap_values_df)
        shap_values_df['Cluster'] = clusters

        # Replace Cluster numbers with A, B, C, D
        cluster_labels = ['A', 'B', 'C', 'D']
        shap_values_df['Cluster'] = shap_values_df['Cluster'].apply(lambda x: cluster_labels[x])

        # Add true labels and prediction results (for a regression task)
        shap_values_df['True_Label'] = true_values

        first_model_key = next(iter(predictions.keys()))
        predicted_values = predictions[first_model_key][:, 0]
        if predicted_values is not None:
            relative_error = np.abs((predicted_values.squeeze() - true_values.squeeze()) / true_values.squeeze()).squeeze()
            shap_values_df['Prediction_Correct'] = (relative_error <= self.threshold).astype(int)
            shap_values_df['Predicted_Label'] = predicted_values

        self.res = {
            "shap_values": shap_values,
            "shap_values_df": shap_values_df,
            "X_test": X_test_np if X_test_np is not None else X_test,
            "feature_names": feature_names,
            "kmeans": kmeans,
            "predictions": predictions
        }

    def _save_to_file(self, path: str, model_name: str) -> None:
        if self.res is None:
            self.logger.warning("No results to save")
            return

        os.makedirs(path, exist_ok=True)

        X_test = self.res["X_test"]
        feature_names = self.res["feature_names"]
        kmeans = self.res["kmeans"]
        # Adjust X_test dimension
        if X_test.ndim == 3:
            X_test = X_test[:, -1, :]

        # Draw the optimized SHAP visualization plot (2x2 matrix)
        self._plot_shap_visualization_combined_optimized(path)

        # Visualize cluster centers
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
        cluster_centers.index = ['A', 'B', 'C', 'D']
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_centers, annot=True, cmap="coolwarm", center=0)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "cluster_centers.svg"), dpi=300)
        plt.close()

        self._plot_model_usage_by_cluster(path)
        self._save_circos_data(path)

    def _plot_shap_visualization_combined_optimized(self, path):
        """
        Combines the SHAP visualization plots of the four clusters into a single 2x2 matrix.
        Removes the main title and ensures subplots are approximately square for better suitability in scientific papers.
        """
        if self.res is None:
            self.logger.warning("No results to plot the combined SHAP graph")
            return

        shap_values_df = self.res["shap_values_df"]
        X_test = self.res["X_test"]
        feature_names = self.res["feature_names"]

        # Take the top 13 features
        shap_values_df_filtered = shap_values_df[feature_names[:13]]

        # Calculate mean(|SHAP value|) and sort
        mean_abs_shap_values = shap_values_df_filtered.abs().mean().sort_values(ascending=False)
        sorted_features = mean_abs_shap_values.index

        # Ensure correct X_test dimension
        if X_test.ndim == 3:
            X_test = X_test[:, -1, :]

        # Build plotting data
        plot_data = []
        for i, row in shap_values_df_filtered.iterrows():
            for j, feature in enumerate(sorted_features):
                original_feature_idx = feature_names.index(feature)
                feature_val = X_test[i, original_feature_idx]

                min_val = X_test[:, original_feature_idx].min()
                max_val = X_test[:, original_feature_idx].max()

                if max_val == min_val:
                    normalized_value = 0.5 
                else:
                    normalized_value = (feature_val - min_val) / (max_val - min_val)
                    
                plot_data.append({
                    "Feature": feature,
                    "SHAP Value": row[feature],
                    "Normalized Value": normalized_value,
                    "Prediction_Correct": shap_values_df.loc[i, "Prediction_Correct"],
                    "Cluster": shap_values_df.loc[i, "Cluster"],
                    "Sample_Index": i
                })

        plot_df = pd.DataFrame(plot_data)

        # ====== Plotting aesthetic parameters ======
        custom_cmap = plt.get_cmap('coolwarm')
        plt.rcParams['font.family'] = 'Times New Roman'
        label_fontsize = 12
        tick_fontsize = 10
        marker_correct = 'o'
        marker_incorrect = 'x'
        s_size = 25
        alpha_val = 0.8
        line_color = '#d3d3d3'
        line_linewidth = 0.8
        line_alpha = 0.6

        # Create a 2x2 subplot with space for the color bar
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.05, 0.475, 0.475], hspace=0.3, wspace=0.2)

        # Draw a shared color bar
        cax = fig.add_subplot(gs[0, :])
        sm = plt.cm.ScalarMappable(cmap=custom_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", fraction=0.04, pad=0.1)
        cbar.ax.set_xlabel("Normalized Feature Value", fontsize=label_fontsize)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        # Unify x-axis range
        all_shap_values = plot_df["SHAP Value"].values
        min_x = all_shap_values.min() - 0.001
        max_x = all_shap_values.max() + 0.001

        cluster_labels = ['A', 'B', 'C', 'D']

        for idx, cluster_id in enumerate(cluster_labels):
            ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])

            cluster_df = plot_df[plot_df["Cluster"] == cluster_id]

            # KDTree for jitter calculation
            all_points = cluster_df["SHAP Value"].values.reshape(-1, 1)
            tree = KDTree(all_points)
            jitter_scale = 0.25
            distance_threshold = 0.005 

            y_pos_map = {feature: i for i, feature in enumerate(sorted_features)}

            for i, feature in enumerate(sorted_features):
                # Key fix: reset_index
                subset = cluster_df[cluster_df["Feature"] == feature].reset_index(drop=True)
                shap_values = subset["SHAP Value"].values
                y_base = np.full(len(shap_values), y_pos_map[feature])
                jitter = np.random.uniform(-jitter_scale, jitter_scale, len(shap_values))

                correct = subset[subset["Prediction_Correct"] == 1]
                incorrect = subset[subset["Prediction_Correct"] == 0]

                # Use zero-based indexing for access
                ax.scatter(
                    correct["SHAP Value"],
                    y_base[correct.index] + jitter[correct.index],
                    c=correct["Normalized Value"],
                    cmap=custom_cmap,
                    s=s_size,
                    alpha=alpha_val,
                    marker=marker_correct,
                    edgecolor='black',
                    linewidth=0.5
                )

                ax.scatter(
                    incorrect["SHAP Value"],
                    y_base[incorrect.index] + jitter[incorrect.index],
                    c=incorrect["Normalized Value"],
                    cmap=custom_cmap,
                    s=s_size,
                    alpha=alpha_val,
                    marker=marker_incorrect,
                    linewidth=0.5
                )

            # Feature trajectory lines
            unique_samples = cluster_df["Sample_Index"].unique()
            for sample_idx in unique_samples:
                if hash(sample_idx) % 5 == 0:
                    sample_data = cluster_df[cluster_df["Sample_Index"] == sample_idx].sort_values(by="Feature", key=lambda x: x.map({f:i for i,f in enumerate(sorted_features)}))
                    shap_values_line = sample_data["SHAP Value"].tolist()
                    
                    if len(shap_values_line) == len(sorted_features):
                        ax.plot(
                            shap_values_line,
                            range(len(sorted_features)),
                            color=line_color,
                            linewidth=line_linewidth,
                            alpha=line_alpha,
                            zorder=0
                        )

            ax.axvline(x=0, color="#808080", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_title(f"Cluster {cluster_id}", fontsize=label_fontsize, fontweight='bold', pad=15)
            ax.set_yticks(range(len(sorted_features)))
            ax.invert_yaxis()
            ax.grid(False)
            ax.set_xlim(min_x, max_x)
            
            # Set x and y axis labels
            if idx % 2 == 0: # Left subplot
                ax.set_ylabel("Features", fontsize=label_fontsize)
                ax.set_yticklabels(sorted_features, fontsize=tick_fontsize)
            else: # Right subplot
                ax.set_yticklabels([])
            
            if idx >= 2: # Bottom subplot
                ax.set_xlabel("SHAP Value", fontsize=label_fontsize)
                ax.tick_params(axis='x', labelsize=tick_fontsize)
            else: # Top subplot
                ax.set_xticklabels([])
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        # Add a legend to explain the meaning of circles and crosses
        legend_elements = [
            plt.Line2D([0], [0], marker=marker_correct, color='black', label='Accurate Prediction',
                       linestyle='', markersize=7, alpha=alpha_val),
            plt.Line2D([0], [0], marker=marker_incorrect, color='black', label='Inaccurate Prediction',
                       linestyle='', markersize=7, alpha=alpha_val)
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=label_fontsize)
        
        plt.savefig(os.path.join(path, "combined_shap_visualization_optimized_2x2.svg"), bbox_inches='tight', dpi=1200)
        plt.close()


    def _plot_model_usage_by_cluster(self, path: str) -> None:
        """
        Plots the prediction status of all clusters on a single graph,
        using different colors to distinguish clusters and shapes to distinguish prediction accuracy.
        """
        if self.res is None:
            self.logger.warning("No results to plot model usage")
            return

        shap_values_df = self.res["shap_values_df"]
        predictions = self.res.get("predictions", {})

        if not predictions:
            self.logger.warning("No model prediction information available, cannot plot model usage")
            return

        # Assuming only one model is provided, take the first one
        model_name = next(iter(predictions.keys()))
        pred_values = predictions[model_name]

        time_steps = np.arange(len(shap_values_df))
        true_values = shap_values_df["True_Label"].values
        clusters = sorted(shap_values_df["Cluster"].unique())

        # Set cluster color mapping
        cluster_colors = {c: plt.cm.Set1(i) for i, c in enumerate(clusters)}

        plt.figure(figsize=(16, 6))

        # Plot true values (global)
        plt.plot(
            time_steps,
            true_values,
            label="True Value",
            color="black",
            linewidth=2,
            linestyle="--",
            zorder=0
        )

        # Plot prediction points for each cluster
        for cluster_id, color in cluster_colors.items():
            cluster_mask = (shap_values_df["Cluster"] == cluster_id)
            correct_mask = shap_values_df.loc[cluster_mask, "Prediction_Correct"] == 1

            # Get the predicted values and time steps for the current cluster
            pred_cluster = pred_values[cluster_mask]
            time_cluster = time_steps[cluster_mask]

            # Plot correctly predicted points
            plt.scatter(
                time_cluster[correct_mask],
                pred_cluster[correct_mask],
                color=color,
                edgecolor='black',
                s=40,
                marker='o',
                alpha=0.8,
                label=f"Cluster {cluster_id} (Correct)",
                zorder=2
            )

            # Plot incorrectly predicted points
            plt.scatter(
                time_cluster[~correct_mask],
                pred_cluster[~correct_mask],
                color=color,
                s=40,
                marker='x',
                alpha=0.8,
                label=f"Cluster {cluster_id} (Incorrect)",
                zorder=2
            )

        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Merge duplicate legend items
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(path, "cluster_based_model_usage.svg"), dpi=300)
        plt.close()