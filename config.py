class Config(object):
    def __init__(self) -> None:
        self.context_size = None
        self.action_size = None
        self.scheduling_mode = None
        self.environment = None
        self.load_weights = None
        self.actor_path = None
        self.critic_path = None
        self.result_path = None
        self.verbose     = None