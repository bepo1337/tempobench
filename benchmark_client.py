import json
import pandas
import benchmark

if __name__ == "__main__":
    # TODO change this to the name of the data/generator
    data_name = "TabSyn"
    # TODO change this line to another path if required to point to your synthetic data
    with open(f"data/synthesized/final_datasets/{data_name}.json") as f:
        synth_data = json.load(f)
        X_syn = pandas.DataFrame(synth_data)

    tempoBench = benchmark.TempoBench(
        X_syn=X_syn,
        # TODO this can be any other name
        generator_name=data_name,
        # TODO comment out any of the categories to exclude them
        metric_categories=[
            benchmark.Category.SANITY,
            benchmark.Category.DOMAIN,
            benchmark.Category.STATISTICAL,
            benchmark.Category.TEMPORAL,
            benchmark.Category.UTILITY
        ],
        # TODO comment out any of the categories to exclude them
        visualization_categories=[
            benchmark.Category.TEMPORAL,
            benchmark.Category.STATISTICAL
        ],
        # TODO change this for the path where the visualizations are stored
        # TODO make sure this directory exists before running TempoBench
        visualization_path = "workspace",
        # TODO change this for the final output path of the json file
        # TODO make sure the directory exists before running TempoBench
        output_path=f"results/{data_name}.json"
    )

    # starts the evaluation
    tempoBench.run()