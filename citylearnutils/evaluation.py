import time
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv


def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict


def get_group_cost(buildings):

    # price_cost
    numer = np.sum([b.net_electricity_consumption_price for b in buildings], 0).clip(0).sum()
    denom = np.sum([b.net_electricity_consumption_without_storage_price for b in buildings], 0).clip(0).sum()
    price_cost = numer / denom

    # emission_cost
    numer = np.sum([b.net_electricity_consumption_emission for b in buildings], 0).sum()
    denom = np.sum([b.net_electricity_consumption_without_storage_emission for b in buildings], 0).sum()
    emission_cost = numer / denom

    # ramp cost
    numer = np.abs(np.diff(np.sum([b.net_electricity_consumption for b in buildings], 0))).sum()
    denom = np.abs(np.diff(np.sum([b.net_electricity_consumption_without_storage for b in buildings], 0))).sum()
    ramp_cost = numer / denom

    # load factor cost
    df = pd.DataFrame(dict(
        x=np.sum([b.net_electricity_consumption for b in buildings], 0),
        y=np.sum([b.net_electricity_consumption_without_storage for b in buildings], 0),
        group=np.arange(len(buildings[0].net_electricity_consumption)) // 730
    )).groupby("group").agg(["mean", "max"])

    numer = np.mean(1 - (df[("x", "mean")] / df[("x", "max")]))
    denom = np.mean(1 - (df[("y", "mean")] / df[("y", "max")]))
    load_factor = numer / denom

    # average_cost
    grid_cost = (ramp_cost + load_factor) / 2
    average_cost = (price_cost + emission_cost + grid_cost) / 3

    return [average_cost, price_cost, emission_cost, grid_cost, ramp_cost, load_factor]


def get_building_costs(buildings):

    building_costs = []
    for i, b in enumerate(buildings):

        # price_cost
        numer = np.sum(b.net_electricity_consumption_price)
        denom = np.sum(b.net_electricity_consumption_without_storage_price)
        price_cost = numer / denom

        # emission_cost
        numer = np.sum(b.net_electricity_consumption_emission)
        denom = np.sum(b.net_electricity_consumption_without_storage_emission)
        emission_cost = numer / denom

        # ramp cost
        numer = np.abs(np.diff(b.net_electricity_consumption)).sum()
        denom = np.abs(np.diff(b.net_electricity_consumption_without_storage)).sum()
        ramp_cost = numer / denom

        # load factor cost
        df = pd.DataFrame(dict(
            x=b.net_electricity_consumption,
            y=b.net_electricity_consumption_without_storage,
            group=np.arange(len(b.net_electricity_consumption)) // 730
        )).groupby("group").agg(["mean", "max"])

        numer = np.mean(1 - (df[("x", "mean")] / df[("x", "max")]))
        denom = np.mean(1 - (df[("y", "mean")] / df[("y", "max")]))
        load_factor = numer / denom

        # average_cost
        grid_cost = (ramp_cost + load_factor) / 2
        average_cost = (price_cost + emission_cost + grid_cost) / 3

        # store results
        building_costs.append((f"building_{i + 1}", average_cost, price_cost, emission_cost,
                               grid_cost, ramp_cost, load_factor))

    return building_costs


def evaluate_agent(agent, n_episodes=1):
    
    env = CityLearnEnv(schema="./data/citylearn_challenge_2022_phase_1/schema.json")

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    num_steps = 0
    episodes_completed = 0
    episode_metrics = []

    while True:

        observations, _, done, _ = env.step(actions)
        num_steps += 1

        if done:

            episodes_completed += 1
            metrics = pd.DataFrame(
                [["public"] + get_group_cost(env.buildings)] +
                get_building_costs(env.buildings),
                columns=["group", "avg", "price", "emission", "grid", "ramp", "load"]
            )

            metrics["episode"] = episodes_completed

            episode_metrics.append(metrics[["episode", "group", "avg", "price", "emission", "grid", "ramp", "load"]])

            if episodes_completed == n_episodes:
                break

            obs_dict = env_reset(env)

            step_start = time.perf_counter()
            num_steps += 1

            agent_time_elapsed += time.perf_counter() - step_start

            actions = agent.register_reset(obs_dict)

        else:
            step_start = time.perf_counter()
            actions = agent.compute_action(observations)
            agent_time_elapsed += time.perf_counter() - step_start

    env_costs = env.evaluate()
    metrics = pd.concat(episode_metrics).sort_index().reset_index(drop=True)

    return env_costs, metrics, agent

