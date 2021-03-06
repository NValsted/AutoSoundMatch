from src.config.registry_sections import GeneticAlgorithmSection

genetic_algorithm_section = GeneticAlgorithmSection(
    out_dim=32,
    population_size=25,
    max_generations=3000,
    crossover_probability=1.0,
    mutation_probability=1.0,
    time_limit=3 * 60,
    patience=200,
)
