import numpy as np
from ..factory.analyzer_factory import AnalyzerFactory
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Add necessary imports for the Taylor diagram
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist import grid_finder

@AnalyzerFactory.register('summary_analyzer')
class SummaryAnalyzer:
    """
    A summary analyzer that reads analysis results from multiple models and
    generates a unified summary report with charts and CSVs.
    The generated charts include MAPE comparison, PICP/PINAW comparison, and Taylor diagrams.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.root_path = self.config.get("root_path", "./res/analysis")
        self.summary_dir = self.config.get("summary_dir", os.path.join(self.root_path, "summary"))
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # --- Global plotting aesthetic settings (SCI paper style) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.color'] = "#636363"
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.8
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 150
        
        # Initialize uniform color and marker maps
        self.model_color_map = {}
        self.model_marker_map = {}

    def analyze(self, predictions_dict, true_values):
        """
        A unified interface that accepts predictions and true_values,
        but SummaryAnalyzer actually ignores them and focuses on root_path.
        """
        print(f"Starting to extract information from {self.root_path} and generate summary report...")
        datasets = self._discover_datasets()

        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            dataset_path = os.path.join(self.root_path, dataset)
            model_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            model_dirs.sort() # Sort models to ensure consistent plotting order

            # Create a dedicated output directory for the current dataset
            dataset_summary_dir = os.path.join(self.summary_dir, dataset)
            os.makedirs(dataset_summary_dir, exist_ok=True)

            # Merge various metrics
            self._merge_metrics_csv(dataset_path, model_dirs, dataset, dataset_summary_dir)
            self._merge_mape_csv(dataset_path, model_dirs, dataset, dataset_summary_dir)
            self._merge_picp_pinaw_csv(dataset_path, model_dirs, dataset, dataset_summary_dir)

            # Generate plots
            self._generate_summary_plots(dataset_path, model_dirs, dataset, dataset_summary_dir)

        print(f"Summary for all datasets is complete. Results saved to {self.summary_dir}")

    def _merge_metrics_csv(self, dataset_path, model_dirs, dataset, dataset_summary_dir):
        all_dfs = []
        for model in model_dirs:
            path = os.path.join(dataset_path, model, f"{model}_metrics.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Model'] = model
                df['Dataset'] = dataset
                all_dfs.append(df)

        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            out_path = os.path.join(dataset_summary_dir, f"{dataset}_performance_comparison.csv")
            merged_df.to_csv(out_path, index=False)
            print(f"Performance comparison CSV for {dataset} generated successfully.")
            
            # Generate Taylor diagram
            self._generate_taylor_diagram(merged_df, dataset_summary_dir, dataset)

    def _merge_mape_csv(self, dataset_path, model_dirs, dataset, dataset_summary_dir):
        all_dfs = []
        for model in model_dirs:
            path = os.path.join(dataset_path, model, "mape_comparison.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Model'] = model
                df['Dataset'] = dataset
                all_dfs.append(df)

        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            out_path = os.path.join(dataset_summary_dir, f"{dataset}_mape_comparison.csv")
            merged_df.to_csv(out_path, index=False)
            print(f"MAPE comparison CSV for {dataset} generated successfully.")

    def _merge_picp_pinaw_csv(self, dataset_path, model_dirs, dataset, dataset_summary_dir):
        all_dfs = []
        for model in model_dirs:
            path = os.path.join(dataset_path, model, "picp_pinaw_comparison.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Model'] = model
                df['Dataset'] = dataset
                all_dfs.append(df)

        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            out_path = os.path.join(dataset_summary_dir, f"{dataset}_picp_pinaw_comparison.csv")
            merged_df.to_csv(out_path, index=False)
            print(f"PICP/PINAW comparison CSV for {dataset} generated successfully.")

    def _generate_summary_plots(self, dataset_path, model_dirs, dataset, dataset_summary_dir):
        mape_csv_path = os.path.join(dataset_summary_dir, f"{dataset}_mape_comparison.csv")
        picp_pinaw_csv_path = os.path.join(dataset_summary_dir, f"{dataset}_picp_pinaw_comparison.csv")
        
        output_pic_dir = os.path.join(dataset_summary_dir, 'pic')

        if os.path.exists(mape_csv_path):
            mape_df = pd.read_csv(mape_csv_path)
            self._plot_mape_results(mape_df, output_pic_dir)

        if os.path.exists(picp_pinaw_csv_path):
            picp_pinaw_df = pd.read_csv(picp_pinaw_csv_path)
            self._plot_picp_pinaw_results(picp_pinaw_df, output_pic_dir)

    def save(self, dataset_name, model_name):
        pass

    def _discover_datasets(self):
        """Discovers all dataset directory names, but skips the summary folder."""
        all_dirs = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
        # Filter out the summary folder
        datasets = [d for d in all_dirs if d != os.path.basename(self.summary_dir)]
        return datasets

    def _plot_mape_results(self, df, output_dir='pic'):
        os.makedirs(output_dir, exist_ok=True)
        # Use a professional seaborn color palette
        colors = sns.color_palette("viridis", 3)
        cities = df['Location'].unique()
        models = df['Model'].unique()
        display_models = [model.split('_')[0] for model in models]

        fig, axes = plt.subplots(nrows=len(cities), ncols=1, figsize=(10, 4.5 * len(cities)), sharey=True)
        if len(cities) == 1:
            axes = [axes]
        
        fig.suptitle('Comparison of MAPE across Different Models and Lead Times', fontsize=14, fontweight='bold')
        fig.subplots_adjust(hspace=0.4, top=0.92)

        for i, city in enumerate(cities):
            ax = axes[i]
            df_city = df[df['Location'] == city]

            bar_width = 0.25
            r1 = np.arange(len(models))
            r2 = r1 + bar_width
            r3 = r2 + bar_width

            # Ensure each model has a corresponding value to prevent indexing errors
            all_results = [df_city[df_city['Model'] == m]['All result'].values[0] if m in df_city['Model'].values else 0 for m in models]
            d1_results = [df_city[df_city['Model'] == m]['D1'].values[0] if m in df_city['Model'].values else 0 for m in models]
            d10_results = [df_city[df_city['Model'] == m]['D10'].values[0] if m in df_city['Model'].values else 0 for m in models]

            bars1 = ax.bar(r1, all_results, width=bar_width, label='All results', color=colors[0], edgecolor='black', zorder=2)
            bars2 = ax.bar(r2, d1_results, width=bar_width, label='D1', color=colors[1], edgecolor='black', zorder=2)
            bars3 = ax.bar(r3, d10_results, width=bar_width, label='D10', color=colors[2], edgecolor='black', zorder=2)

            ax.set_title(f'Location: {city}', fontsize=12, pad=10)
            ax.set_ylabel('MAPE (%)', fontsize=11)
            ax.set_xticks(r2)
            ax.set_xticklabels(display_models)
            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
            ax.legend(fontsize=9, loc='upper right')

            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0: # Avoid displaying labels for values of 0
                         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.2f}',
                                 ha='center', va='bottom', fontsize=8, rotation=45) # Rotate labels to avoid overlap
            
            add_labels(bars1)
            add_labels(bars2)
            add_labels(bars3)
            
        plt.savefig(os.path.join(output_dir, 'mape_comparison.svg'), bbox_inches='tight')
        plt.close()

    def _plot_picp_pinaw_results(self, df, output_dir='pic'):
        os.makedirs(output_dir, exist_ok=True)
        locations = df['Location'].unique()
        
        models = df['Model'].unique()
        display_models = [model.split('_')[0] for model in models]
        colors = sns.color_palette("Paired", len(models))
        x_pos = np.arange(len(models))

        for location in locations:
            loc_df = df[df['Location'] == location].copy()
            
            summary_df = loc_df.groupby('Model').agg({
                'PICP': ['mean', 'std'],
                'PINAW': ['mean', 'std']
            }).reset_index()
            summary_df.columns = ['Model', 'PICP_mean', 'PICP_std', 'PINAW_mean', 'PINAW_std']
            
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Prediction Interval Performance for {location}', fontsize=14, fontweight='bold')
            fig.subplots_adjust(top=0.92, hspace=0.4)

            # --- PICP subplot ---
            ax1.set_title('Prediction Interval Coverage Probability (PICP)', fontsize=12)
            ax1.set_ylabel('PICP', fontsize=11)
            ax1.set_ylim(0, 1.1)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            bars1 = ax1.bar(x_pos, summary_df['PICP_mean'], yerr=summary_df['PICP_std'],
                             width=0.6, capsize=5, zorder=2,
                             color=colors, edgecolor='black', align='center')

            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}',
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='Target PICP (95%)', zorder=1)
            ax1.legend(loc='lower right', fontsize=9)

            # --- PINAW subplot ---
            ax2.set_title('Prediction Interval Normalized Average Width (PINAW)', fontsize=12)
            ax2.set_ylabel('PINAW', fontsize=11)
            ax2.set_xlabel('Models', fontsize=12)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(display_models, rotation=30, ha='right')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            bars2 = ax2.bar(x_pos, summary_df['PINAW_mean'], yerr=summary_df['PINAW_std'],
                             width=0.6, capsize=5, zorder=2,
                             color=colors, edgecolor='black', align='center')

            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}',
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.savefig(os.path.join(output_dir, f'picp_pinaw_comparison_{location}.svg'), bbox_inches='tight')
            plt.close()

    def _generate_taylor_diagram(self, df, output_dir, dataset_name):
        """
        Generates a single Taylor diagram plotting data points for all models.
        """
        os.makedirs(os.path.join(output_dir, 'pic'), exist_ok=True)
        
        # Get all model names and establish a uniform color and marker mapping
        models = df['Model'].unique()
        display_models = sorted([model.split('_')[0] for model in models]) # Sort to ensure consistent order
        
        if not self.model_color_map or not self.model_marker_map:
            # Use a more professional color palette and markers
            colors = sns.color_palette("viridis", len(display_models))
            markers = ['o', 's', '^', 'D', 'p', 'h', 'v', '>', '<', '*', 'X', 'P']
            for i, display_name in enumerate(display_models):
                self.model_color_map[display_name] = colors[i]
                self.model_marker_map[display_name] = markers[i % len(markers)]
        
        # Create a single figure and plot data for all models on it
        fig = plt.figure(figsize=(8, 8))
        taylor_ax = self._setup_taylor_axes(fig, 111)
        
        if len(df) > 0:
            self._plot_taylor_data_on_axes(taylor_ax, df)
        
        # Set the main title for the Taylor diagram
        taylor_ax.set_title(f'Taylor Diagram for {dataset_name}', fontsize=14, fontweight='bold', pad=20)
        
        # Add a legend (with de-duplication)
        handles, labels = taylor_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # --- Adjust legend position to be closer to the plot ---
        taylor_ax.legend(by_label.values(), by_label.keys(), 
                         loc='upper left',
                         bbox_to_anchor=(0.85, 1.0),
                         fontsize=9, 
                         frameon=True, fancybox=True, shadow=True,
                         handletextpad=0.5)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'pic', f'{dataset_name}_taylor_diagram_combined.svg')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Unified Taylor diagram generated successfully: {output_path}")

    def _setup_taylor_axes(self, fig, location):
        """
        Sets up the Taylor diagram axes, optimizing layout and labels.
        """
        trans = PolarAxes.PolarTransform()

        # Set the positions and labels for the first grid (correlation coefficient)
        r1_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
        t1_locs = np.arccos(r1_locs)
        gl1 = grid_finder.FixedLocator(t1_locs)
        tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str, r1_locs))))

        # Set the positions and labels for the second grid (standard deviation)
        r2_locs = np.arange(0, 2, 0.25)
        gl2 = grid_finder.FixedLocator(r2_locs)
        tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str, r2_locs))))

        # Create helper objects for the polar coordinate system
        ghelper = floating_axes.GridHelperCurveLinear(trans, extremes=(0, np.pi/2, 0, 1.75),
                                                     grid_locator1=gl1, tick_formatter1=tf1,
                                                     grid_locator2=gl2, tick_formatter2=tf2)
        ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Set properties and labels for the polar axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation Coefficient")
        ax.axis["top"].label.set_fontsize(11)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation (Normalized)")
        ax.axis["left"].label.set_fontsize(11)
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        
        # Set tick label font sizes
        ax.axis["top"].major_ticklabels.set_fontsize(8)
        ax.axis["right"].major_ticklabels.set_fontsize(8)
        ax.axis["left"].major_ticklabels.set_fontsize(8)

        ax.grid(True, linestyle='-', alpha=0.6, color='gray', linewidth=0.6)

        # Get the auxiliary axes object for the polar plot
        polar_ax = ax.get_aux_axes(trans)
        polar_ax.patch.set_facecolor('white')

        # Create contours for the Taylor diagram (RMSD)
        rs, ts = np.meshgrid(np.linspace(0, 1.75, 100),
                             np.linspace(0, np.pi/2, 100))
        rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
        CS = polar_ax.contour(ts, rs, rms, levels=np.arange(0, 2.1, 0.25), 
                              colors='#2E8B57', linestyles='--', linewidths=0.8, alpha=0.7)
        plt.clabel(CS, inline=1, fontsize=8, fmt='%.2f', colors='black')

        # Plot the reference line (circle with Std Dev = 1)
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + 1
        polar_ax.plot(t, r, 'k--', linewidth=1)

        # Plot the reference point
        polar_ax.plot(0, 1, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1, 
                      label="Reference", zorder=5)
        polar_ax.text(0.05, 1.1, "Reference", size=9, ha="left", va="bottom", 
                      bbox=dict(boxstyle="round,pad=0.2", ec='red', fc='white', alpha=0.8),
                      fontweight='bold', color='red')

        return polar_ax

    def _plot_taylor_data_on_axes(self, axes, df):
        """
        Plots data points on the Taylor diagram using a uniform color map.
        """
        # Get the reference standard deviation (Std Dev of true values)
        ref_std = df['RefStdDev'].mean() if 'RefStdDev' in df.columns else 1.0
        
        # Sort the DataFrame to ensure the legend order matches the plotting order
        df_sorted = df.sort_values(by='Model')
        
        for idx, row in df_sorted.iterrows():
            # Calculate normalized standard deviation
            std_dev = row['StdDev'] / ref_std if ref_std != 0 else row['StdDev']
            
            # Calculate angle (based on correlation coefficient)
            correlation = max(-1, min(1, row['Correlation']))
            theta = np.arccos(correlation)
            
            # Use the uniform color and marker
            model_name = row['Model']
            display_name = model_name.split('_')[0]
            
            color = self.model_color_map.get(display_name, 'black')
            marker = self.model_marker_map.get(display_name, 'o')
            
            # Plot the data point
            axes.plot(theta, std_dev, marker=marker, markersize=9, 
                      color=color, label=display_name, alpha=0.9,
                      markeredgecolor='black', markeredgewidth=0.8)