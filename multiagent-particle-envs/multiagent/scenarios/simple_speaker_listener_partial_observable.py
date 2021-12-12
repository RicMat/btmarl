import numpy as np
from multiagent.scenarios.simple_speaker_listener import Scenario
from multiagent.scenarios.scenario_util import obscure_pos


def distance(p1, p2):
    vector = p2 - p1
    d = np.sqrt(np.sum(np.square(vector)))
    return d


class POScenario(Scenario):
    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append((agent.state.p_pos- entity.state.p_pos))

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            if distance(agent.state.p_pos, entity.state.p_pos) < 0.5:
                comm.append(other.state.c)
            else:
                comm.append([0,0,0])

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
