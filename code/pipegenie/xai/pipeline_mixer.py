from itertools import permutations, combinations
from pipegenie.pipeline import Pipeline
from pipegenie.evolutionary._individual import Individual
from pipegenie.xai.pipeline_registry import PipelineRegistry

class PipelineMixer:
    def __init__(self, ind_uuid: str, registry: PipelineRegistry):
        self.ind_uuid = ind_uuid
        self.registry = registry
        self.ind_data = registry.get(ind_uuid)

    def permutations_pipeline(self) -> 'list[Individual]':
        ind_perms = []
        ind_steps = self.ind_data.steps
        all_steps_mix = []
        if len(ind_steps) > 1:
            all_steps_mix = [list(p) + [ind_steps[-1]] for p in permutations(ind_steps[:-1])]

        for steps in all_steps_mix:
            new_ind = Individual(content=self.ind_data.content,
                                 fitness=self.ind_data.fitness)
            new_ind.pipeline = Pipeline(steps=steps)
            ind_perms.append(new_ind)

        return ind_perms
    
    def combinations_pipeline(self) -> 'list[Individual]':
        ind_combs = []
        ind_steps = self.ind_data.steps
    
        all_ind_bin = []
        if len(ind_steps) > 1:
            base = ind_steps[:-1]
            fixed = ind_steps[-1]
            for r in range(0, len(base)):
                for comb in combinations(base, r):
                    all_ind_bin.append(list(comb) + [fixed])
        
        for steps in all_ind_bin:
            print(steps)
            new_ind = Individual(content=self.ind_data.content)
            new_ind.pipeline = Pipeline(steps=steps)
            new_ind.add_parent(self.ind_uuid)
            old_ind_str = self.ind_data.ind_str.split(";")
            new_ind_str = ';'.join(old_ind_str[int(i)] for i, _ in steps)
            new_ind._cached_str = new_ind_str
            ind_combs.append(new_ind)

        return ind_combs