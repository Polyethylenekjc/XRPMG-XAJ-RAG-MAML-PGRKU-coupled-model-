class BaseFactory:
    _registry = {}

    @classmethod
    def register(cls, name=None):
        """Decorator method to register classes"""
        def decorator(subclass):
            """The actual decorator function"""
            cls._registry[name or subclass.__name__] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, class_name, *args, **kwargs):
        """Create a class instance by name, supporting arbitrary parameter passing"""
        if class_name not in cls._registry:
            raise ValueError(f'Class {class_name} is not registered.')
        return cls._registry[class_name](*args, **kwargs)

    @classmethod
    def get_registered_classes(cls):
        """Get a list of all registered class names"""
        return list(cls._registry.keys())

    @classmethod
    def create_from_config(cls, config_key, config):
        """Create an instance from a configuration dictionary"""
        class_name = config.get('type')
        params = config.get('params', {})
        if not class_name:
            raise ValueError(f'Missing required key "type" in config for {config_key}')
        return cls.create(class_name, **params)

    @classmethod
    def create_all(cls, config_list):
        """
        Bulk create instances from a list of configurations

        Args:
            config_list: A list of configurations containing type names and parameters

        Returns:
            A list of created instances
        """
        instances = []
        for item in config_list:
            name = item.get('name')
            params = item.get('params', {})
            instance = cls.create(name, **params)
            instances.append(instance)
        return instances