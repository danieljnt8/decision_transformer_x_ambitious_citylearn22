from gym.spaces import Box


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


class OrderEnforcingAgent:
    """
    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """

    def __init__(self, agent):
        self.num_buildings = None
        self.agent = agent
        self.action_space = None

    def register_reset(self, obs):
        """Get the first observation after env.reset, return action"""
        action_space = obs["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        observation_list = obs["observation"]
        self.num_buildings = len(observation_list)

        for agent_id, observation in enumerate(observation_list):
            action_space = self.action_space[agent_id]
            self.agent.set_action_space(agent_id, action_space)
            self.agent.reset_agent(agent_id, observation)

        return self.compute_action(observation_list)

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, observation_list):
        """
        Inputs:
            observation - List of observations from the env
        Returns:
            actions - List of actions in the same order as the observations

        You can change this function as needed
        please make sure the actions are in same order as the observations

        Reward preprocesing - You can use your custom reward function here
        please specify your reward function in agents/user_agent.py

        """
        return self.agent.compute_action(observation_list)


