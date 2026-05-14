# -------------
# Add the import
# -------------
import uuid

class Individual(SyntaxTreePipeline):
    # -------------
    # Update the init method
    # -------------
    def __init__(
        self,
        content: list['Union[TerminalNode, NonTerminalNode]'],
        fitness: 'Optional[Fitness]' = None,
    ):
        super().__init__(content)
        self.uuid = uuid.uuid4()
        self.parents_uuid = []
        self.content = content
        self.fitness = fitness if fitness is not None else Fitness((1.0,))
        self.diversity: float = np.nan
        self.prediction: 'NDArray' = np.array([])

    # -------------
    # Add the add_parent method
    # -------------
    def add_parent(self, parent: uuid) -> None:
        if len(self.parents_uuid) == 2:
            pass
        elif len(self.parents_uuid) < 2:
            self.parents_uuid.append(parent)

    # -------------
    # Update the reset method
    # -------------
    def reset(self) -> None:
        """
        Reset the individual.
        """
        super().reset()
        self.fitness.invalidate()
        self.uuid = uuid.uuid4()
