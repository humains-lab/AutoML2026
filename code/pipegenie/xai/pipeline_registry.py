from typing import TYPE_CHECKING
from pipegenie.evolutionary._individual import Individual

if TYPE_CHECKING:
    from typing import Union
    from pipegenie.evolutionary._individual import Individual

class PipelineRegistry:
    def __init__(self):
        self._pipelines = {}

    def register(self, uuid, pipeline_metrics: 'MetricsPipeline'):
        self._pipelines[uuid] = pipeline_metrics

    def get(self, uuid):
        return self._pipelines[uuid]

    def all(self):
        return self._pipelines.values()

class MetricsPipeline():
    def __init__(self, ind: Individual):
        self.ind = ind.clone()
        self.ind_str = str(ind)
        self.ind_parents = ind.parents_uuid
        self.content = ind.content
        self.fitness = ind.fitness
        self.runtime = ind.runtime
        self.hp_number = self._hp_counter(ind)
        self.steps = ind.pipeline.steps
        self.families = self._step_families(ind)
    
    def get_dict(self):
        return {
            "id": str(self.ind.uuid),
            "parents_id": [str(parent_id) for parent_id in self.ind_parents],
            "pipeline": self.ind_str,
            "fitness": list(self.fitness.values),
            "runtime": self.runtime,
            "hp_number": self.hp_number,
            "families": self.families
        }

    def _extract_parentheses(self, s: str):
        stack = []
        pairs = []

        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            elif c == ")":
                start = stack.pop()
                pairs.append((start, i))

        results = []

        for start, end in pairs:
            content = s[start + 1:end]

            parts = []
            current = []
            depth = 0

            for ch in content:
                if ch == "(":
                    depth += 1
                    current.append(ch)
                elif ch == ")":
                    depth -= 1
                    current.append(ch)
                elif ch == "," and depth == 0:
                    parts.append("".join(current))
                    current = []
                else:
                    current.append(ch)

            if current:
                parts.append("".join(current))

            results.append(parts)

        return results

    def _hp_counter(self, ind: 'Union[Individual,str]') -> int:
        hps_by_algorithm = self._extract_parentheses(str(ind))
        hp_counter = 0
        for hps in hps_by_algorithm:
            hp_counter += len(hps)
        return hp_counter

    def _step_families(self, ind: Individual) -> 'list[str]':
        return [step[0] for step in ind.pipeline.steps]
    

registry = PipelineRegistry()
