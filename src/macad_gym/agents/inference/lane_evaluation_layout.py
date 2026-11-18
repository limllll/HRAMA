"""Evaluation-only lane-change NPC selection; shared training scene is untouched."""


def apply_lane_npc_count(configs, npc_count):
    if npc_count not in (2, 3):
        raise ValueError('npc_count must be 2 or 3')
    expected = {'ego', 'car1', 'car2', 'car3'}
    actors = configs['actors']
    scenario = configs['scenarios']
    if not isinstance(scenario, dict) or scenario.get('map') != 'Town04':
        raise ValueError('Expected the resolved Town04 lane-change scenario')
    scenario_actors = scenario['actors']
    if set(actors) != expected or set(scenario_actors) != expected:
        raise ValueError('Expected matching ego/car1/car2/car3 configurations')
    if npc_count == 3:
        return []
    # Match the user's existing BH 2NPC layout: remove the front vehicle
    # in ego's target lane, not the rear car1 or same-lane lead car3.
    del actors['car2']
    del scenario_actors['car2']
    return ['car2']
