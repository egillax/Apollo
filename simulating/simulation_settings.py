from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationModelSettings:
    dynamic_state_count: int
    fixed_state_count: int
    concept_count: int
    serious_concept_count: int
    visit_multiplier: int
    days_to_simulate: int

    def __init__(self, config: ConfigParser):
        self.dynamic_state_count = config.getint("simulation", "dynamic_state_count")
        self.fixed_state_count = config.getint("simulation", "fixed_state_count")
        self.concept_count = config.getint("simulation", "concept_count")
        self.serious_concept_count = config.getint("simulation", "serious_concept_count")
        self.visit_multiplier = config.getint("simulation", "visit_multiplier")
        self.days_to_simulate = config.getint("simulation", "days_to_simulate")


@dataclass
class SimulationSettings:
    # system
    max_cores: int
    root_folder: str

    # pre - training data generation
    generate_pre_training_data: bool
    partition_count: int
    person_count: int

    # prediction data generation
    generate_prediction_data: bool
    partition_count: int
    train_person_count: int
    test_person_count: int
    prediction_window: int

    # simulation
    json_file_name: Optional[str]
    simulation_model_settings: Optional[SimulationModelSettings]

    # debug
    profile: bool
    log_verbosity: int

    def __init__(self, config: ConfigParser):
        self.max_cores = config.getint("system", "max_cores")
        self.root_folder = config.get("system", "root_folder")

        self.generate_pre_training_data = config.getboolean("pre_training_data_generation",
                                                            "generate_pre_training_data")
        self.partition_count = config.getint("pre_training_data_generation", "partition_count")
        self.person_count = config.getint("pre_training_data_generation", "person_count")

        self.generate_prediction_data = config.getboolean("prediction_data_generation", "generate_prediction_data")
        self.partition_count = config.getint("prediction_data_generation", "partition_count")
        self.train_person_count = config.getint("prediction_data_generation", "train_person_count")
        self.test_person_count = config.getint("prediction_data_generation", "test_person_count")
        self.prediction_window = config.getint("prediction_data_generation", "prediction_window")

        self.json_file_name = config.get("simulation", "json_file_name")
        if self.json_file_name.strip() == "":
            self.json_file_name = None
            self.simulation_model_settings = SimulationModelSettings(config)
        else:
            self.simulation_model_settings = None

        self.profile = config.getboolean("debug", "profile")
        self.log_verbosity = config.getint("debug", "log_verbosity")

        self._validate()

    def _validate(self):
        if self.log_verbosity not in [0, 1, 2]:
            raise ValueError(f"Invalid log_verbosity: {self.log_verbosity}")
