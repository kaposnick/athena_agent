class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.num_episodes_inference = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False
        self.save_results = False
        self.results_file_path = None
        self.save_weights = False
        self.save_weights_file = None
        self.save_weights_period = 1e3
        self.load_initial_weights = False
        self.initial_weights_path = None