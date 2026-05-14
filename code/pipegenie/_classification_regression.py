# -------------
# Add the import
# -------------
from pipegenie.xai.pipeline_registry import MetricsPipeline

class BaseClassificationRegression(BasePipegenie, ABC):
    # -------------
    # Update the _generate_ensemble method
    # -------------
    def _generate_ensemble(self, data: 'dict[str, ArrayLike]') -> None:
        X = data["X"]
        y = data["y"]

        best_ind = self.elite.best_ind()
        best_ind.pipeline.fit(X, y)
        best_fitness = best_ind.fitness.values[0]

        weights = [float(ind.fitness.values[0] / best_fitness) if self.maximization
                   else float(best_fitness / ind.fitness.values[0])
                   for ind in self.elite]
        
        valid_elite = [ind for ind in self.elite if ind.fitness.valid]
        if len(valid_elite) > 0:
            for elite in valid_elite:
                self.registry.register(elite.uuid, MetricsPipeline(elite))

            estimators = [(str(idx), est.pipeline) for idx, est in enumerate(valid_elite)]
            self.ensemble = self._create_ensemble_object(estimators, weights)
            # Reset ramdon state
            self.sampler_validator.set_random_state(self.seed)
            fitness_values = []
            
            # Evaluate the ensemble with the cross-validator
            for train_idx, test_idx in self.sampler_validator.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                self.ensemble.fit(X_train, y_train)
                y_pred = self.ensemble.predict(X_test)

                fitness = self.fitness(y_test, y_pred)
                fitness_values.append(fitness)

            # Retrain the ensemble with the whole dataset
            self.ensemble.fit(X, y)

            with self.outdir_path.joinpath("best_pipeline.txt").open("w", encoding="utf-8") as log:
                log.write(str(best_ind) + "\n")
                log.write(f"Fitness: {best_fitness}\n")
                log.write(f"Prediction: {best_ind.pipeline.predict(X)}\n")

            with self.outdir_path.joinpath("ensemble.txt").open("w", encoding="utf-8") as log:
                for idx, (name, est) in enumerate(estimators):
                    log.write(f"{name}: {est} -> Fitness: {self.elite[idx].fitness.values[0]}\n\n")

                log.write(f"Ensemble fitness: {np.mean(fitness_values)}\n")
                log.write(f"Weights: {weights}\n")
                log.write(f"Prediction: {self.ensemble.predict(X)}\n")

        else:
            with self.outdir_path.joinpath("best_pipeline.txt").open("w", encoding="utf-8") as log:
                log.write("")
            with self.outdir_path.joinpath("ensemble.txt").open("w", encoding="utf-8") as log:
                log.write("")