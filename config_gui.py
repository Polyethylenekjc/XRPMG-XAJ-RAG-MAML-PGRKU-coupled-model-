# config_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os


class ConfigManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GTLF Config Manager")
        self.root.geometry("1100x750")
        self.root.minsize(1000, 700)
        
        # Configuration file path
        self.config_file = "/home/fifth/code/Python/GTLF/Config/test.yaml"
        
        # Current configuration data
        self.config_data = {}
        
        # Shared parameters
        self.shared_params = {
            "seq_length": tk.IntVar(value=15),
            "forecast_steps": tk.IntVar(value=3),
            "input_dim": tk.IntVar(value=10),
            "forecast_horizon": tk.IntVar(value=1) # Added new variable for forecast_horizon
        }
        
        # Current selected module
        self.current_module = tk.StringVar(value="model")
        
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="GTLF Configuration Manager", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15), anchor=tk.W)
        
        # Create left-right layout frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left navigation bar
        nav_frame = ttk.LabelFrame(content_frame, text="Navigation", width=150)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        nav_frame.pack_propagate(False)
        
        # Navigation buttons
        modules = [
            ("Models", "model"),
            ("Datasets", "data"),
            ("Analyzers", "analyzer"),
            ("Shared Parameters", "shared")
        ]
        
        for text, module in modules:
            btn = ttk.Button(nav_frame, text=text, 
                            command=lambda m=module: self.switch_module(m))
            btn.pack(fill=tk.X, pady=2, padx=5)
        
        # Right content area
        self.content_area = ttk.Frame(content_frame)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # File operation area
        file_frame = ttk.Frame(self.content_area)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(file_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(10, 0))
        self.file_path_label = ttk.Label(file_frame, text=self.config_file)
        self.file_path_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Module content area
        self.module_content_frame = ttk.Frame(self.content_area)
        self.module_content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize display of first module
        self.switch_module("model")
        
    def switch_module(self, module):
        self.current_module.set(module)
        
        # Clear current content
        for widget in self.module_content_frame.winfo_children():
            widget.destroy()
            
        # Display content based on selected module
        if module == "model":
            self.show_model_config()
        elif module == "data":
            self.show_data_config()
        elif module == "analyzer":
            self.show_analyzer_config()
        elif module == "shared":
            self.show_shared_config()
            
    def show_model_config(self):
        # Model configuration area
        model_frame = ttk.LabelFrame(self.module_content_frame, text="Model Configuration", padding="15")
        model_frame.pack(fill=tk.BOTH, expand=True)
        model_frame.columnconfigure(0, weight=1)
        model_frame.rowconfigure(0, weight=1)
        
        # Create model tabs
        self.model_notebook = ttk.Notebook(model_frame)
        self.model_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Update model tabs
        self.update_model_tabs()
        
    def show_data_config(self):
        # Dataset configuration area
        dataset_frame = ttk.LabelFrame(self.module_content_frame, text="Dataset Configuration", padding="15")
        dataset_frame.pack(fill=tk.BOTH, expand=True)
        dataset_frame.columnconfigure(0, weight=1)
        dataset_frame.rowconfigure(3, weight=1)
        
        # Dataset operation buttons
        dataset_button_frame = ttk.Frame(dataset_frame)
        dataset_button_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        ttk.Button(dataset_button_frame, text="Add Dataset", command=self.add_dataset).pack(side=tk.LEFT)
        ttk.Button(dataset_button_frame, text="Edit Dataset", command=self.edit_dataset).pack(side=tk.LEFT, padx=(10, 10))
        ttk.Button(dataset_button_frame, text="Remove Dataset", command=self.remove_dataset).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(dataset_button_frame, text="Enable Selected", command=self.enable_selected_datasets).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(dataset_button_frame, text="Disable Selected", command=self.disable_selected_datasets).pack(side=tk.LEFT)
        ttk.Button(dataset_button_frame, text="Unified Processors", command=self.unified_processor_config).pack(side=tk.LEFT, padx=(10, 0))
        
        # Dataset list headers
        list_header_frame = ttk.Frame(dataset_frame)
        list_header_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(list_header_frame, text="Status", font=("Arial", 9, "bold"), width=10).pack(side=tk.LEFT)
        ttk.Label(list_header_frame, text="Dataset Name", font=("Arial", 9, "bold"), width=20).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(list_header_frame, text="Path", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        
        # Dataset list
        list_frame = ttk.Frame(dataset_frame)
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Create styled dataset list with multi-selection support
        self.dataset_tree = ttk.Treeview(list_frame, columns=("status", "name", "path"), show="headings", height=12, selectmode="extended")
        self.dataset_tree.heading("status", text="Status")
        self.dataset_tree.heading("name", text="Dataset Name")
        self.dataset_tree.heading("path", text="Path")
        self.dataset_tree.column("status", width=80, anchor=tk.CENTER)
        self.dataset_tree.column("name", width=150, anchor=tk.W)
        self.dataset_tree.column("path", width=400, anchor=tk.W)
        
        self.dataset_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_tree.configure(yscrollcommand=scrollbar.set)
        
        # Dataset details area
        detail_frame = ttk.LabelFrame(dataset_frame, text="Dataset Details", padding="10")
        detail_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        detail_frame.columnconfigure(1, weight=1)
        
        self.dataset_detail_text = tk.Text(detail_frame, height=8, state=tk.DISABLED)
        self.dataset_detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.dataset_tree.bind("<<TreeviewSelect>>", self.on_dataset_select)
        
        # Update dataset list
        self.update_dataset_list()
        
    def show_analyzer_config(self):
        # Analyzer configuration area
        analyzer_frame = ttk.LabelFrame(self.module_content_frame, text="Analyzer Configuration", padding="15")
        analyzer_frame.pack(fill=tk.BOTH, expand=True)
        analyzer_frame.columnconfigure(0, weight=1)
        analyzer_frame.rowconfigure(2, weight=1)
        
        # Analyzer operation buttons
        analyzer_button_frame = ttk.Frame(analyzer_frame)
        analyzer_button_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        ttk.Button(analyzer_button_frame, text="Add Analyzer", command=self.add_analyzer).pack(side=tk.LEFT)
        ttk.Button(analyzer_button_frame, text="Edit Analyzer", command=self.edit_analyzer).pack(side=tk.LEFT, padx=(10, 10))
        ttk.Button(analyzer_button_frame, text="Remove Analyzer", command=self.remove_analyzer).pack(side=tk.LEFT, padx=(0, 10))
        
        # Analyzer list headers
        list_header_frame = ttk.Frame(analyzer_frame)
        list_header_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(list_header_frame, text="Name", font=("Arial", 9, "bold"), width=20).pack(side=tk.LEFT)
        ttk.Label(list_header_frame, text="Parameters", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        
        # Analyzer list
        list_frame = ttk.Frame(analyzer_frame)
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Create analyzer list
        self.analyzer_tree = ttk.Treeview(list_frame, columns=("name", "params"), show="headings", height=12)
        self.analyzer_tree.heading("name", text="Analyzer Name")
        self.analyzer_tree.heading("params", text="Parameters")
        self.analyzer_tree.column("name", width=200, anchor=tk.W)
        self.analyzer_tree.column("params", width=400, anchor=tk.W)
        
        self.analyzer_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.analyzer_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analyzer_tree.configure(yscrollcommand=scrollbar.set)
        
        # Analyzer details area
        detail_frame = ttk.LabelFrame(analyzer_frame, text="Analyzer Details", padding="10")
        detail_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        detail_frame.columnconfigure(1, weight=1)
        
        self.analyzer_detail_text = tk.Text(detail_frame, height=8, state=tk.DISABLED)
        self.analyzer_detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.analyzer_tree.bind("<<TreeviewSelect>>", self.on_analyzer_select)
        
        # Update analyzer list
        self.update_analyzer_list()
        
    def show_shared_config(self):
        # Shared parameters area
        shared_frame = ttk.LabelFrame(self.module_content_frame, text="Shared Parameters", padding="15")
        shared_frame.pack(fill=tk.BOTH, expand=True)
        
        # First row
        row_frame = ttk.Frame(shared_frame)
        row_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row_frame, text="Sequence Length:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=self.shared_params["seq_length"], width=12).pack(side=tk.LEFT, padx=(10, 30))
        
        ttk.Label(row_frame, text="Forecast Steps:").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Entry(row_frame, textvariable=self.shared_params["forecast_steps"], width=12).pack(side=tk.LEFT, padx=(10, 0))
        
        # Second row
        row_frame = ttk.Frame(shared_frame)
        row_frame.pack(fill=tk.X, pady=(15, 5))
        
        ttk.Label(row_frame, text="Input Dimension:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=self.shared_params["input_dim"], width=12).pack(side=tk.LEFT, padx=(10, 30))
        
        ttk.Button(row_frame, text="Apply to All Models", command=self.apply_shared_params).pack(side=tk.LEFT, padx=(20, 0))
        
        # New row for forecast_horizon
        row_frame = ttk.Frame(shared_frame)
        row_frame.pack(fill=tk.X, pady=(15, 5))
        
        ttk.Label(row_frame, text="Forecast Horizon:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=self.shared_params["forecast_horizon"], width=12).pack(side=tk.LEFT, padx=(10, 30))
        
        ttk.Button(row_frame, text="Apply to Config", command=self.update_all_forecast_horizons).pack(side=tk.LEFT, padx=(20, 0))
        
    def on_dataset_select(self, event):
        selection = self.dataset_tree.selection()
        if selection:
            item = self.dataset_tree.item(selection[0])
            dataset_name = item['values'][1]
            
            # Find selected dataset
            dataset = None
            if 'data' in self.config_data and 'datasets' in self.config_data['data']:
                for d in self.config_data['data']['datasets']:
                    if d['name'] == dataset_name:
                        dataset = d
                        break
            
            if dataset:
                # Display details
                self.dataset_detail_text.config(state=tk.NORMAL)
                self.dataset_detail_text.delete(1.0, tk.END)
                
                detail_text = f"Name: {dataset.get('name', 'N/A')}\n"
                detail_text += f"Path: {dataset.get('path', 'N/A')}\n"
                detail_text += f"Enabled: {dataset.get('enabled', True)}\n"
                detail_text += f"Target Column: {dataset.get('target_column', 'N/A')}\n"
                detail_text += f"Test Size: {dataset.get('test_size', 'N/A')}\n"
                detail_text += f"Window: {dataset.get('window', 'N/A')}\n"
                
                if 'processors' in dataset:
                    detail_text += f"Processors: {len(dataset['processors'])} items\n"
                    for i, processor in enumerate(dataset['processors']):
                        detail_text += f"  {i+1}. {processor.get('name', 'N/A')}"
                        if 'params' in processor and processor['params']:
                            params_str = ', '.join([f"{k}={v}" for k, v in processor['params'].items()])
                            detail_text += f" ({params_str})"
                        detail_text += "\n"
                
                self.dataset_detail_text.insert(tk.END, detail_text)
                self.dataset_detail_text.config(state=tk.DISABLED)
        
    def on_analyzer_select(self, event):
        selection = self.analyzer_tree.selection()
        if selection:
            item = self.analyzer_tree.item(selection[0])
            analyzer_name = item['values'][0]
            
            # Find selected analyzer
            analyzer = None
            if 'analyzers' in self.config_data:
                for a in self.config_data['analyzers']:
                    if a['name'] == analyzer_name:
                        analyzer = a
                        break
            
            if analyzer:
                # Display details
                self.analyzer_detail_text.config(state=tk.NORMAL)
                self.analyzer_detail_text.delete(1.0, tk.END)
                
                detail_text = f"Name: {analyzer.get('name', 'N/A')}\n"
                
                if 'params' in analyzer and analyzer['params']:
                    detail_text += "Parameters:\n"
                    for key, value in analyzer['params'].items():
                        detail_text += f"  {key}: {value}\n"
                else:
                    detail_text += "Parameters: None\n"
                
                self.analyzer_detail_text.insert(tk.END, detail_text)
                self.analyzer_detail_text.config(state=tk.DISABLED)
        
    def load_config(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
            
            # Update shared parameters
            if 'models' in self.config_data and self.config_data['models']:
                first_model = list(self.config_data['models'].keys())[0]
                model_params = self.config_data['models'][first_model]['params']
                
                if 'seq_length' in model_params:
                    self.shared_params['seq_length'].set(model_params['seq_length'])
                if 'forecast_steps' in model_params or 'output_dim' in model_params:
                    forecast_value = model_params.get('forecast_steps') or model_params.get('output_dim')
                    self.shared_params['forecast_steps'].set(forecast_value)
                if 'input_dim' in model_params:
                    self.shared_params['input_dim'].set(model_params['input_dim'])

            # Update new forecast_horizon variable
            if 'forecast_horizon' in self.config_data:
                self.shared_params['forecast_horizon'].set(self.config_data['forecast_horizon'])
            
            # Update currently displayed module
            self.switch_module(self.current_module.get())
            
            messagebox.showinfo("Success", "Configuration file loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration file: {str(e)}")
            
    def save_config(self):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, allow_unicode=True, default_flow_style=False, indent=2)
            messagebox.showinfo("Success", "Configuration file saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration file: {str(e)}")
            
    def update_model_tabs(self):
        # Clear existing tabs
        for tab in self.model_notebook.tabs():
            self.model_notebook.forget(tab)
            
        # Create tab for each model
        if 'models' not in self.config_data:
            self.config_data['models'] = {}
            
        for model_name, model_config in self.config_data['models'].items():
            tab = ttk.Frame(self.model_notebook, padding="10")
            self.model_notebook.add(tab, text=model_name)
            tab.columnconfigure(0, weight=1)
            
            # Enable status
            enabled_var = tk.BooleanVar(value=model_config.get('enabled', False))
            enable_frame = ttk.Frame(tab)
            enable_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
            ttk.Checkbutton(enable_frame, text="Enable Model", variable=enabled_var, 
                           command=lambda mn=model_name, var=enabled_var: self.update_model_enabled(mn, var)).pack(side=tk.LEFT)
            
            # Create left-right column frames
            content_frame = ttk.Frame(tab)
            content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            content_frame.columnconfigure((0, 1), weight=1)
            
            # Model parameters
            params_frame = ttk.LabelFrame(content_frame, text="Model Parameters", padding="15")
            params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10), pady=(0, 15))
            params_frame.columnconfigure(1, weight=1)
            
            row = 0
            if 'params' in model_config:
                for param_name, param_value in model_config['params'].items():
                    ttk.Label(params_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, pady=3)
                    param_var = tk.StringVar(value=str(param_value))
                    entry = ttk.Entry(params_frame, textvariable=param_var, width=20)
                    entry.grid(row=row, column=1, padx=(10, 0), pady=3, sticky=(tk.W, tk.E))
                    entry.bind('<FocusOut>', lambda e, mn=model_name, pn=param_name, pv=param_var: 
                              self.update_model_param(mn, pn, pv.get()))
                    row += 1
                
            # Trainer parameters
            trainer_frame = ttk.LabelFrame(content_frame, text="Trainer Parameters", padding="15")
            trainer_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=(10, 0), pady=(0, 15))
            trainer_frame.columnconfigure(1, weight=1)
            
            row = 0
            if 'trainer' in model_config:
                for param_name, param_value in model_config['trainer'].items():
                    ttk.Label(trainer_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, pady=3)
                    param_var = tk.StringVar(value=str(param_value))
                    entry = ttk.Entry(trainer_frame, textvariable=param_var, width=20)
                    entry.grid(row=row, column=1, padx=(10, 0), pady=3, sticky=(tk.W, tk.E))
                    entry.bind('<FocusOut>', lambda e, mn=model_name, pn=param_name, pv=param_var: 
                              self.update_trainer_param(mn, pn, pv.get()))
                    row += 1
            
    def update_model_enabled(self, model_name, enabled_var):
        if 'models' in self.config_data and model_name in self.config_data['models']:
            self.config_data['models'][model_name]['enabled'] = enabled_var.get()
        
    def update_model_param(self, model_name, param_name, param_value):
        # Try to convert to appropriate data type
        converted_value = self.convert_value(param_value)
        if 'models' in self.config_data and model_name in self.config_data['models']:
            if 'params' in self.config_data['models'][model_name]:
                self.config_data['models'][model_name]['params'][param_name] = converted_value
        
    def update_trainer_param(self, model_name, param_name, param_value):
        # Try to convert to appropriate data type
        converted_value = self.convert_value(param_value)
        if 'models' in self.config_data and model_name in self.config_data['models']:
            if 'trainer' in self.config_data['models'][model_name]:
                self.config_data['models'][model_name]['trainer'][param_name] = converted_value
        
    def convert_value(self, value):
        # Try to convert to number or boolean
        if isinstance(value, str):
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value  # Keep as string
        return value
            
    def apply_shared_params(self):
        seq_length = self.shared_params['seq_length'].get()
        forecast_steps = self.shared_params['forecast_steps'].get()
        input_dim = self.shared_params['input_dim'].get()
        
        if 'models' not in self.config_data:
            self.config_data['models'] = {}
            
        for model_name, model_config in self.config_data['models'].items():
            if 'params' not in model_config:
                model_config['params'] = {}
                
            params = model_config['params']
            
            # Update shared parameters
            if 'seq_length' in params:
                params['seq_length'] = seq_length
            if 'input_dim' in params:
                params['input_dim'] = input_dim
                
            # Update forecast parameters based on model type
            if 'grku' in model_name.lower() and 'output_dim' in params:
                params['output_dim'] = forecast_steps
            elif 'forecast_steps' in params:
                params['forecast_steps'] = forecast_steps
                
        # Update UI
        self.update_model_tabs()
        messagebox.showinfo("Success", "Shared parameters applied to all models")
        
    def update_all_forecast_horizons(self):
        """
        Update the forecast_horizon in the main config and all model trainers.
        """
        try:
            forecast_horizon = self.shared_params['forecast_horizon'].get()
            
            # Update the global forecast_horizon
            self.config_data['forecast_horizon'] = forecast_horizon
            
            # Update the forecast_horizon in each model's trainer
            if 'models' in self.config_data:
                for model_config in self.config_data['models'].values():
                    if 'trainer' in model_config:
                        model_config['trainer']['forecast_horizon'] = forecast_horizon
            
            self.update_model_tabs() # Refresh the UI to show changes
            messagebox.showinfo("Success", f"Forecast horizon updated to {forecast_horizon} in all relevant locations.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update forecast horizon: {str(e)}")

    def update_dataset_list(self):
        # Clear existing items
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
            
        if 'data' not in self.config_data:
            self.config_data['data'] = {'datasets': []}
        elif 'datasets' not in self.config_data['data']:
            self.config_data['data']['datasets'] = []
            
        datasets = self.config_data['data']['datasets']
        for dataset in datasets:
            status = "Enabled" if dataset.get('enabled', True) else "Disabled"
            name = dataset.get('name', 'Unknown')
            path = dataset.get('path', 'N/A')
            self.dataset_tree.insert("", tk.END, values=(status, name, path))
            
    def update_analyzer_list(self):
        # Clear existing items
        for item in self.analyzer_tree.get_children():
            self.analyzer_tree.delete(item)
            
        if 'analyzers' not in self.config_data:
            self.config_data['analyzers'] = []
            
        analyzers = self.config_data['analyzers']
        for analyzer in analyzers:
            name = analyzer.get('name', 'Unknown')
            params_info = "None"
            if 'params' in analyzer and analyzer['params']:
                params_info = ", ".join([f"{k}={v}" for k, v in list(analyzer['params'].items())[:3]])
                if len(analyzer['params']) > 3:
                    params_info += "..."
            self.analyzer_tree.insert("", tk.END, values=(name, params_info))
            
    def enable_selected_datasets(self):
        """
        Enable selected datasets
        """
        selections = self.dataset_tree.selection()
        if not selections:
            messagebox.showwarning("Warning", "Please select at least one dataset to enable")
            return
            
        if 'data' in self.config_data and 'datasets' in self.config_data['data']:
            datasets = self.config_data['data']['datasets']
            changed_count = 0
            
            for selection in selections:
                item = self.dataset_tree.item(selection)
                dataset_name = item['values'][1]
                
                for dataset in datasets:
                    if dataset['name'] == dataset_name and not dataset.get('enabled', True):
                        dataset['enabled'] = True
                        changed_count += 1
                        
            if changed_count > 0:
                self.update_dataset_list()
                messagebox.showinfo("Success", f"Enabled {changed_count} dataset(s)")
            else:
                messagebox.showinfo("Info", "Selected dataset(s) are already enabled")
        
    def disable_selected_datasets(self):
        """
        Disable selected datasets
        """
        selections = self.dataset_tree.selection()
        if not selections:
            messagebox.showwarning("Warning", "Please select at least one dataset to disable")
            return
            
        if 'data' in self.config_data and 'datasets' in self.config_data['data']:
            datasets = self.config_data['data']['datasets']
            changed_count = 0
            
            for selection in selections:
                item = self.dataset_tree.item(selection)
                dataset_name = item['values'][1]
                
                for dataset in datasets:
                    if dataset['name'] == dataset_name and dataset.get('enabled', True):
                        dataset['enabled'] = False
                        changed_count += 1
                        
            if changed_count > 0:
                self.update_dataset_list()
                messagebox.showinfo("Success", f"Disabled {changed_count} dataset(s)")
            else:
                messagebox.showinfo("Info", "Selected dataset(s) are already disabled")
        
    def toggle_dataset_enable(self):
        selection = self.dataset_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to toggle")
            return
            
        item = self.dataset_tree.item(selection[0])
        dataset_name = item['values'][1]
        
        if 'data' in self.config_data and 'datasets' in self.config_data['data']:
            datasets = self.config_data['data']['datasets']
            for dataset in datasets:
                if dataset['name'] == dataset_name:
                    current_status = dataset.get('enabled', True)
                    dataset['enabled'] = not current_status
                    
                    self.update_dataset_list()
                    status_text = "enabled" if not current_status else "disabled"
                    messagebox.showinfo("Success", f"Dataset {status_text}")
                    return
        
    def add_dataset(self):
        dialog = DatasetDialog(self.root, "Add Dataset")
        if dialog.result:
            if 'data' not in self.config_data:
                self.config_data['data'] = {'datasets': []}
            elif 'datasets' not in self.config_data['data']:
                self.config_data['data']['datasets'] = []
                
            new_dataset = {
                'name': dialog.result['name'],
                'path': dialog.result['path'],
                'enabled': dialog.result.get('enabled', True)
            }
            
            # Add other parameters
            for key, value in dialog.result.items():
                if key not in ['name', 'path', 'enabled']:
                    new_dataset[key] = value
                    
            self.config_data['data']['datasets'].append(new_dataset)
            self.update_dataset_list()
            messagebox.showinfo("Success", "Dataset added successfully")
            
    def add_analyzer(self):
        dialog = AnalyzerDialog(self.root, "Add Analyzer")
        if dialog.result:
            if 'analyzers' not in self.config_data:
                self.config_data['analyzers'] = []
                
            self.config_data['analyzers'].append(dialog.result)
            self.update_analyzer_list()
            messagebox.showinfo("Success", "Analyzer added successfully")
        
    def edit_dataset(self):
        selection = self.dataset_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to edit")
            return
            
        item = self.dataset_tree.item(selection[0])
        dataset_name = item['values'][1]
        
        if 'data' not in self.config_data or 'datasets' not in self.config_data['data']:
            messagebox.showerror("Error", "No datasets available")
            return
            
        datasets = self.config_data['data']['datasets']
        dataset = None
        for d in datasets:
            if d['name'] == dataset_name:
                dataset = d
                break
                
        if dataset:
            dialog = DatasetDialog(self.root, "Edit Dataset", dataset)
            if dialog.result:
                # Update dataset
                for i, d in enumerate(datasets):
                    if d['name'] == dataset_name:
                        datasets[i] = dialog.result
                        break
                self.update_dataset_list()
                messagebox.showinfo("Success", "Dataset updated successfully")
        else:
            messagebox.showerror("Error", "Invalid dataset selection")
            
    def edit_analyzer(self):
        selection = self.analyzer_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an analyzer to edit")
            return
            
        item = self.analyzer_tree.item(selection[0])
        analyzer_name = item['values'][0]
        
        if 'analyzers' not in self.config_data:
            messagebox.showerror("Error", "No analyzers available")
            return
            
        analyzers = self.config_data['analyzers']
        analyzer = None
        for a in analyzers:
            if a['name'] == analyzer_name:
                analyzer = a
                break
                
        if analyzer:
            dialog = AnalyzerDialog(self.root, "Edit Analyzer", analyzer)
            if dialog.result:
                # Update analyzer
                for i, a in enumerate(analyzers):
                    if a['name'] == analyzer_name:
                        analyzers[i] = dialog.result
                        break
                self.update_analyzer_list()
                messagebox.showinfo("Success", "Analyzer updated successfully")
        else:
            messagebox.showerror("Error", "Invalid analyzer selection")
        
    def remove_dataset(self):
        selection = self.dataset_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to remove")
            return
            
        item = self.dataset_tree.item(selection[0])
        dataset_name = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to remove dataset '{dataset_name}'?"):
            if 'data' in self.config_data and 'datasets' in self.config_data['data']:
                datasets = self.config_data['data']['datasets']
                for i, dataset in enumerate(datasets):
                    if dataset['name'] == dataset_name:
                        del datasets[i]
                        self.update_dataset_list()
                        messagebox.showinfo("Success", "Dataset removed successfully")
                        return
                    
    def remove_analyzer(self):
        selection = self.analyzer_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an analyzer to remove")
            return
            
        item = self.analyzer_tree.item(selection[0])
        analyzer_name = item['values'][0]
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to remove analyzer '{analyzer_name}'?"):
            if 'analyzers' in self.config_data:
                analyzers = self.config_data['analyzers']
                for i, analyzer in enumerate(analyzers):
                    if analyzer['name'] == analyzer_name:
                        del analyzers[i]
                        self.update_analyzer_list()
                        messagebox.showinfo("Success", "Analyzer removed successfully")
                        return

    def unified_processor_config(self):
        """Unified configuration for all dataset processors"""
        # Get current processors from config (if any)
        current_processors = []
        if 'data' in self.config_data and 'default_dataset' in self.config_data['data']:
            default_dataset = self.config_data['data']['default_dataset']
            if 'processors' in default_dataset:
                current_processors = default_dataset['processors']
        
        dialog = ProcessorsDialog(self.root, "Unified Processor Configuration", current_processors)
        if dialog.result is not None:
            # Update default dataset configuration
            if 'data' not in self.config_data:
                self.config_data['data'] = {}
            if 'default_dataset' not in self.config_data['data']:
                self.config_data['data']['default_dataset'] = {}
            
            self.config_data['data']['default_dataset']['processors'] = dialog.result
            
            # Apply to all datasets
            if 'datasets' in self.config_data['data']:
                for dataset in self.config_data['data']['datasets']:
                    dataset['processors'] = dialog.result
            
            # Update UI
            self.update_dataset_list()
            messagebox.showinfo("Success", "Unified processor configuration applied to all datasets")


class DatasetDialog:
    def __init__(self, parent, title, dataset=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("500x450")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center dialog
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.top.winfo_screenheight() // 2) - (450 // 2)
        self.top.geometry(f"500x450+{x}+{y}")
        
        self.result = None
        
        # Variables
        self.name_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=True)
        self.target_column_var = tk.StringVar(value="runoffs")
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.window_var = tk.IntVar(value=15)
        
        # If in edit mode, populate existing data
        if dataset:
            self.name_var.set(dataset.get('name', ''))
            self.path_var.set(dataset.get('path', ''))
            self.enabled_var.set(dataset.get('enabled', True))
            self.target_column_var.set(dataset.get('target_column', 'runoffs'))
            self.test_size_var.set(dataset.get('test_size', 0.2))
            self.window_var.set(dataset.get('window', 15))
        
        self.setup_ui()
        parent.wait_window(self.top)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Dataset Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Form frame
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name
        name_frame = ttk.Frame(form_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(name_frame, text="Name:").pack(anchor=tk.W)
        ttk.Entry(name_frame, textvariable=self.name_var).pack(fill=tk.X, pady=(5, 0))
        
        # Path
        path_frame = ttk.Frame(form_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(path_frame, text="Path:").pack(anchor=tk.W)
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(path_entry_frame, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(path_entry_frame, text="Browse", command=self.browse_path, width=8).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Target column
        target_frame = ttk.Frame(form_frame)
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="Target Column:").pack(anchor=tk.W)
        ttk.Entry(target_frame, textvariable=self.target_column_var).pack(fill=tk.X, pady=(5, 0))
        
        # Test size
        test_size_frame = ttk.Frame(form_frame)
        test_size_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(test_size_frame, text="Test Size:").pack(anchor=tk.W)
        ttk.Entry(test_size_frame, textvariable=self.test_size_var).pack(fill=tk.X, pady=(5, 0))
        
        # Window size
        window_frame = ttk.Frame(form_frame)
        window_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(window_frame, text="Window Size:").pack(anchor=tk.W)
        ttk.Entry(window_frame, textvariable=self.window_var).pack(fill=tk.X, pady=(5, 0))
        
        # Enable status
        enable_frame = ttk.Frame(form_frame)
        enable_frame.pack(fill=tk.X, pady=(0, 25))
        
        ttk.Checkbutton(enable_frame, text="Enable Dataset", variable=self.enabled_var).pack(anchor=tk.W)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.top.bind('<Return>', lambda e: self.ok())
        self.top.bind('<Escape>', lambda e: self.cancel())
    
    def browse_path(self):
        path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("All files", "*.*"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("YAML files", "*.yaml *.yml")
            ]
        )
        if path:
            self.path_var.set(path)
    
    def ok(self):
        if not self.name_var.get().strip():
            messagebox.showerror("Error", "Dataset name is required", parent=self.top)
            return
            
        self.result = {
            'name': self.name_var.get().strip(),
            'path': self.path_var.get().strip(),
            'enabled': self.enabled_var.get(),
            'target_column': self.target_column_var.get().strip(),
            'test_size': self.test_size_var.get(),
            'window': self.window_var.get()
        }
        self.top.destroy()
    
    def cancel(self):
        self.top.destroy()


class AnalyzerDialog:
    def __init__(self, parent, title, analyzer=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("550x450")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center dialog
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (550 // 2)
        y = (self.top.winfo_screenheight() // 2) - (450 // 2)
        self.top.geometry(f"550x450+{x}+{y}")
        
        self.result = None
        self.params = {}
        
        # Variables
        self.name_var = tk.StringVar()
        
        # If in edit mode, populate existing data
        if analyzer:
            self.name_var.set(analyzer.get('name', ''))
            if 'params' in analyzer:
                self.params = analyzer['params'].copy()
        
        self.setup_ui()
        parent.wait_window(self.top)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Analyzer Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Form frame
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name
        name_frame = ttk.Frame(form_frame)
        name_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(name_frame, text="Name:").pack(anchor=tk.W)
        ttk.Entry(name_frame, textvariable=self.name_var).pack(fill=tk.X, pady=(5, 0))
        
        # Parameters area
        params_frame = ttk.LabelFrame(form_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Parameter buttons
        params_btn_frame = ttk.Frame(params_frame)
        params_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(params_btn_frame, text="Add Parameter", command=self.add_parameter).pack(side=tk.LEFT)
        ttk.Button(params_btn_frame, text="Edit Parameter", command=self.edit_parameter).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(params_btn_frame, text="Remove Parameter", command=self.remove_parameter).pack(side=tk.LEFT, padx=(10, 0))
        
        # Parameter list
        list_frame = ttk.Frame(params_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.params_listbox = tk.Listbox(list_frame, height=10)
        self.params_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.params_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Initialize parameter list
        self.update_params_list()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.top.bind('<Return>', lambda e: self.ok())
        self.top.bind('<Escape>', lambda e: self.cancel())
        
    def update_params_list(self):
        self.params_listbox.delete(0, tk.END)
        for key, value in self.params.items():
            self.params_listbox.insert(tk.END, f"{key}: {value}")
    
    def add_parameter(self):
        dialog = ParameterDialog(self.top, "Add Parameter")
        if dialog.result:
            key, value = dialog.result
            self.params[key] = value
            self.update_params_list()
    
    def edit_parameter(self):
        selection = self.params_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a parameter to edit", parent=self.top)
            return
        
        index = selection[0]
        key = list(self.params.keys())[index]
        value = self.params[key]
        
        dialog = ParameterDialog(self.top, "Edit Parameter", key, value)
        if dialog.result:
            new_key, new_value = dialog.result
            # Delete old key-value pair
            del self.params[key]
            # Add new key-value pair
            self.params[new_key] = new_value
            self.update_params_list()
    
    def remove_parameter(self):
        selection = self.params_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a parameter to remove", parent=self.top)
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to remove this parameter?", parent=self.top):
            index = selection[0]
            key = list(self.params.keys())[index]
            del self.params[key]
            self.update_params_list()
    
    def ok(self):
        if not self.name_var.get().strip():
            messagebox.showerror("Error", "Analyzer name is required", parent=self.top)
            return
            
        self.result = {
            'name': self.name_var.get().strip()
        }
        
        if self.params:
            self.result['params'] = self.params
            
        self.top.destroy()
    
    def cancel(self):
        self.top.destroy()


class ParameterDialog:
    def __init__(self, parent, title, key=None, value=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("400x150")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center dialog
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.top.winfo_screenheight() // 2) - (150 // 2)
        self.top.geometry(f"400x150+{x}+{y}")
        
        self.result = None
        
        # Variables
        self.key_var = tk.StringVar()
        self.value_var = tk.StringVar()
        
        # If in edit mode, populate existing data
        if key is not None and value is not None:
            self.key_var.set(key)
            self.value_var.set(str(value))
        
        self.setup_ui()
        parent.wait_window(self.top)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Form
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Key
        key_frame = ttk.Frame(form_frame)
        key_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(key_frame, text="Key:", width=10).pack(side=tk.LEFT)
        ttk.Entry(key_frame, textvariable=self.key_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Value
        value_frame = ttk.Frame(form_frame)
        value_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(value_frame, text="Value:", width=10).pack(side=tk.LEFT)
        ttk.Entry(value_frame, textvariable=self.value_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.top.bind('<Return>', lambda e: self.ok())
        self.top.bind('<Escape>', lambda e: self.cancel())
    
    def ok(self):
        if not self.key_var.get().strip():
            messagebox.showerror("Error", "Parameter key is required", parent=self.top)
            return
            
        # Try to convert value to appropriate data type
        value_str = self.value_var.get().strip()
        value = value_str
        
        if value_str.lower() in ['true', 'false']:
            value = value_str.lower() == 'true'
        else:
            try:
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                pass  # Keep as string
        
        self.result = (self.key_var.get().strip(), value)
        self.top.destroy()
    
    def cancel(self):
        self.top.destroy()


class ProcessorsDialog:
    def __init__(self, parent, title, processors=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("600x400")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center dialog
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.top.winfo_screenheight() // 2) - (400 // 2)
        self.top.geometry(f"600x400+{x}+{y}")
        
        self.result = None
        self.processors = processors if processors else []
        
        self.setup_ui()
        parent.wait_window(self.top)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Processor Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Info text
        info_label = ttk.Label(main_frame, text="Configuration will be applied to all datasets", font=("Arial", 10))
        info_label.pack(pady=(0, 10))
        
        # Processor operation buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="Add Processor", command=self.add_processor).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Edit Processor", command=self.edit_processor).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(btn_frame, text="Remove Processor", command=self.remove_processor).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(btn_frame, text="Move Up", command=self.move_up).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(btn_frame, text="Move Down", command=self.move_down).pack(side=tk.LEFT, padx=(10, 0))
        
        # Processor list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.processors_listbox = tk.Listbox(list_frame, height=12)
        self.processors_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.processors_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.processors_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Initialize processor list
        self.update_processors_list()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.top.bind('<Return>', lambda e: self.ok())
        self.top.bind('<Escape>', lambda e: self.cancel())
        
        # Bind double-click to edit
        self.processors_listbox.bind('<Double-Button-1>', lambda e: self.edit_processor())
    
    def update_processors_list(self):
        self.processors_listbox.delete(0, tk.END)
        for i, processor in enumerate(self.processors):
            name = processor.get('name', 'Unknown')
            params = processor.get('params', {})
            if params:
                params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                self.processors_listbox.insert(tk.END, f"{i+1}. {name} ({params_str})")
            else:
                self.processors_listbox.insert(tk.END, f"{i+1}. {name}")
    
    def add_processor(self):
        dialog = ProcessorEditDialog(self.top, "Add Processor")
        if dialog.result:
            self.processors.append(dialog.result)
            self.update_processors_list()
    
    def edit_processor(self):
        selection = self.processors_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a processor to edit", parent=self.top)
            return
        
        index = selection[0]
        processor = self.processors[index]
        
        dialog = ProcessorEditDialog(self.top, "Edit Processor", processor)
        if dialog.result:
            self.processors[index] = dialog.result
            self.update_processors_list()
    
    def remove_processor(self):
        selection = self.processors_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a processor to remove", parent=self.top)
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to remove the selected processor?", parent=self.top):
            index = selection[0]
            del self.processors[index]
            self.update_processors_list()
    
    def move_up(self):
        selection = self.processors_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a processor to move", parent=self.top)
            return
        
        index = selection[0]
        if index > 0:
            self.processors[index], self.processors[index-1] = self.processors[index-1], self.processors[index]
            self.update_processors_list()
            self.processors_listbox.selection_set(index-1)
    
    def move_down(self):
        selection = self.processors_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a processor to move", parent=self.top)
            return
        
        index = selection[0]
        if index < len(self.processors) - 1:
            self.processors[index], self.processors[index+1] = self.processors[index+1], self.processors[index]
            self.update_processors_list()
            self.processors_listbox.selection_set(index+1)
    
    def ok(self):
        self.result = self.processors
        self.top.destroy()
    
    def cancel(self):
        self.top.destroy()


class ProcessorEditDialog:
    def __init__(self, parent, title, processor=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("450x350")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center dialog
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.top.winfo_screenheight() // 2) - (350 // 2)
        self.top.geometry(f"450x350+{x}+{y}")
        
        self.result = None
        
        # Variables
        self.name_var = tk.StringVar()
        self.params = {}
        
        # If in edit mode, populate existing data
        if processor:
            self.name_var.set(processor.get('name', ''))
            if 'params' in processor:
                self.params = processor['params'].copy()
        
        self.setup_ui()
        parent.wait_window(self.top)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Processor Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Name
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(name_frame, text="Name:").pack(anchor=tk.W)
        ttk.Entry(name_frame, textvariable=self.name_var).pack(fill=tk.X, pady=(5, 0))
        
        # Parameters area
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Parameter buttons
        params_btn_frame = ttk.Frame(params_frame)
        params_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(params_btn_frame, text="Add Parameter", command=self.add_parameter).pack(side=tk.LEFT)
        ttk.Button(params_btn_frame, text="Edit Parameter", command=self.edit_parameter).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(params_btn_frame, text="Remove Parameter", command=self.remove_parameter).pack(side=tk.LEFT, padx=(10, 0))
        
        # Parameter list
        list_frame = ttk.Frame(params_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.params_listbox = tk.Listbox(list_frame, height=6)
        self.params_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.params_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Initialize parameter list
        self.update_params_list()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.top.bind('<Return>', lambda e: self.ok())
        self.top.bind('<Escape>', lambda e: self.cancel())
        
        # Bind double-click to edit
        self.params_listbox.bind('<Double-Button-1>', lambda e: self.edit_parameter())
    
    def update_params_list(self):
        self.params_listbox.delete(0, tk.END)
        for key, value in self.params.items():
            self.params_listbox.insert(tk.END, f"{key}: {value}")
    
    def add_parameter(self):
        dialog = ParameterDialog(self.top, "Add Parameter")
        if dialog.result:
            key, value = dialog.result
            self.params[key] = value
            self.update_params_list()
    
    def edit_parameter(self):
        selection = self.params_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a parameter to edit", parent=self.top)
            return
        
        index = selection[0]
        key = list(self.params.keys())[index]
        value = self.params[key]
        
        dialog = ParameterDialog(self.top, "Edit Parameter", key, value)
        if dialog.result:
            new_key, new_value = dialog.result
            # Delete old key-value pair
            del self.params[key]
            # Add new key-value pair
            self.params[new_key] = new_value
            self.update_params_list()
    
    def remove_parameter(self):
        selection = self.params_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a parameter to remove", parent=self.top)
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to remove the selected parameter?", parent=self.top):
            index = selection[0]
            key = list(self.params.keys())[index]
            del self.params[key]
            self.update_params_list()
    
    def ok(self):
        if not self.name_var.get().strip():
            messagebox.showerror("Error", "Processor name is required", parent=self.top)
            return
            
        self.result = {
            'name': self.name_var.get().strip()
        }
        
        if self.params:
            self.result['params'] = self.params
        else:
            self.result['params'] = {}
            
        self.top.destroy()
    
    def cancel(self):
        self.top.destroy()


def main():
    root = tk.Tk()
    app = ConfigManagerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()