import yaml

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
    
    @classmethod
    def from_config(cls, config):
        config_path = config.get('config_path')
        if not config_path:
            raise ValueError("Missing 'config_path' in configuration")
        return cls(config_path=config_path)
    
    def load_config(self):
        """Loads and validates the configuration file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate that required fields exist
            if 'data' not in config or 'datasets' not in config['data']:
                raise ValueError("Invalid config format: missing 'data.datasets' field")
            
            if 'models' not in config:
                raise ValueError("Invalid config format: missing 'models' field")
            
            if 'analyzers' not in config:
                raise ValueError("Invalid config format: missing 'analyzers' field")
                
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def get_config(self):
        """Gets the complete configuration"""
        return self.config
    
    def get_model_config(self):
        """Gets the model configuration, returns an empty dictionary if it doesn't exist"""
        return self.config.get('models', {})
    
    def get_data_processor_config(self):
        """Gets the data processor configuration, returns an empty dictionary if it doesn't exist"""
        return self.config.get('data_processors', {})
    
    def get_analyzer_config(self):
        """Gets the analyzer configuration"""
        return self.config.get('analyzers', {})
    
    def get_output_config(self):
        """Gets the output configuration"""
        return self.config.get('output', {})
