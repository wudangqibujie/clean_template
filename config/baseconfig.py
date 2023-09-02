import yaml


class BaseConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def from_cmdline(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            return cls(data)

    def from_dict(self):
        pass
