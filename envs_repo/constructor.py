from envs_repo.gym_wrapper import GymWrapper

class EnvConstructor:
    #daytradeEnv
    def __init__(self, env_name, frameskip):
        self.env_name = env_name
        self.frameskip = frameskip

        daytrade = self.make_env()
        self.is_discrete = daytrade.is_discrete
        self.state_dim = daytrade.state_dim
        self.action_dim = daytrade.action_dim

    def make_env(self, **kwargs):
        env = GymWrapper(self.env_name, self.frameskip)
        return env  
