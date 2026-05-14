# -------------
# Add the imports
# -------------
import json
from pipegenie.xai.pipeline_registry import PipelineRegistry, MetricsPipeline
from pipegenie.xai.pipeline_mixer import PipelineMixer
from pipegenie.evolutionary._individual import Fitness

class BasePipegenie(ABC):
    # -------------
    # Update the init method
    # -------------
    def __init__(
        self,
        grammar: str,
        *,
        grammar_type: str,
        pop_size: int,
        generations: int,
        fitness: 'Callable[..., object]',
        nderiv: int,
        selection: 'SelectionBase',
        crossover: 'CrossoverBase',
        mutation: 'MutationBase',
        mutation_elite: 'MutationBase',
        replacement: 'ReplacementBase',
        use_double_mutation: bool,
        elite_size: int,
        timeout: int,
        eval_timeout: 'Optional[int]',
        sampler_validator: 'BaseSamplerValidator',
        maximization: bool,
        early_stopping_threshold: float,
        early_stopping_generations: 'Optional[int]',
        early_stopping_time: 'Optional[int]',
        seed: 'Optional[int]',
        outdir: str,
        n_jobs: int,
        **kwargs: object,
    ):
        self.grammar = grammar
        self.grammar_type = grammar_type
        self.pop_size = pop_size
        self.generations = generations
        self.fitness = fitness
        self.nderiv = nderiv
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.mutation_elite = mutation_elite
        self.replacement = replacement
        self.use_double_mutation = use_double_mutation
        self.elite_size = elite_size
        self.timeout = timeout
        self.eval_timeout = eval_timeout
        self.sampler_validator = sampler_validator
        self.maximization = maximization
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_time = early_stopping_time
        self.outdir = outdir
        self.n_jobs = n_jobs

        self.seed = seed if seed is not None else random.randint(0, 2**32)
        random.seed(self.seed)

        # Ensure that the cross-validator is initialized with the same seed
        self.sampler_validator.set_random_state(random_state=self.seed)

        arguments = vars(self).copy()
        arguments.update(kwargs)

        self.outdir_path = Path(self.outdir)
        self.outdir_path.mkdir(parents=True, exist_ok=True)

        self.cpu_count = cpu_count() if self.n_jobs == -1 else self.n_jobs

        # log the configuration
        with self.outdir_path.joinpath("config.txt").open("w", encoding="utf-8") as log:
            for key, value in arguments.items():
                if hasattr(value, '__name__'):
                    log.write(f"{key}: {value.__name__}\n")
                else:
                    log.write(f"{key}: {value}\n")

        self._create_loggers()
        self._init_statistics()

        self.general_logger.info("PipeGenie initiated")

        root, terms, non_terms, self.pset, _ = parse_pipe_grammar(
            grammar,
            grammar_type,
            self.seed,
        )
        self.schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)
        self.registry = PipelineRegistry()
    

    # -------------
    # Add the ensemble_explainer method
    # -------------
    def ensemble_explainer(self, X: 'ArrayLike', y: 'ArrayLike') -> None:
        if y is None or len(y) == 0:
            raise ValueError("The target values are missing")

        if X is None or len(X) == 0 or (X is not None and len(X) != len(y)):
            raise ValueError("The training input samples are missing or have an invalid shape")

        X_copy = np.array(X)
        y_copy = np.array(y)

        if y_copy.ndim > 1:
            if y_copy.shape[1] == 1:
                y_copy = y_copy.ravel()
            else:
                raise ValueError("The 'y' parameter should be unidimensional")

        data = {
            "X": X_copy,
            "y": y_copy,
        }

        valid_elite = [ind for ind in self.elite if ind.fitness.valid]
        if len(valid_elite) > 0:

            start = time()
            all_comb_elites = []
            for elite in valid_elite:
                comb_elite = PipelineMixer(elite.uuid, self.registry).combinations_pipeline()
                all_comb_elites += comb_elite

            chunksize = 1 if self.cpu_count == 1 else ceil((self.pop_size / self.cpu_count) * 0.25)

            manager = Manager()
            q = manager.Queue()

            explainer = partial(self._explainer, data=data, start=start, queue=q)

            with ProcessPoolExecutor(max_workers=self.cpu_count) as pool:
                results = pool.map(explainer, all_comb_elites, chunksize=chunksize)

            for ind, result in zip(all_comb_elites, results, strict=True):
                ind.fitness.values, ind.prediction, ind.runtime = result
                self.registry.register(ind.uuid, MetricsPipeline(ind))
        
        with self.outdir_path.joinpath("registry_data.json").open("w", encoding="utf-8") as log:
            registry_data = self.registry.all()
            data_list = []
            for each_data in registry_data:
                data_list.append(each_data.get_dict())
            json.dump(data_list, log)


    # -------------
    # Add the _explainer method
    # -------------
    def _explainer(
        self,
        ind: 'Individual',
        data: 'dict[str, ArrayLike]',
        start: float,
        queue: 'Queue',
    ) -> 'tuple[float, Optional[NDArray], Optional[float]]':
        if "X" not in data or "y" not in data:
            raise ValueError("Missing evaluation data. Either 'X' or 'y' is missing.")

        if ind.fitness.valid:
            return ind.fitness.values, ind.prediction, ind.runtime

        if (time() - start) > self.timeout:
            return (np.nan,), None, None

        start_eval = time()
        fitness, predictions = self._evaluate_cv(ind, data["X"], data["y"])
        elapsed_time = time() - start_eval

        if isinstance(fitness, str):
            return (np.nan,), None, None

        return (fitness,), predictions, elapsed_time