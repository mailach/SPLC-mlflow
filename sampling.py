import sys
from caching import CacheHandler
import pandas as pd
from rich.logging import RichHandler
import logging
import mlflow
import click

from splc2py.sampling import Sampler


def activate_logging(logs_to_artifact):
    if logs_to_artifact:
        return logging.basicConfig(
            filename="logs.txt",
            level=logging.INFO,
            format="SAMPLING    %(message)s",
        )
    return logging.basicConfig(
        level=logging.INFO,
        format="SAMPLING    %(message)s",
        handlers=[RichHandler()],
    )


def _extract_parameters(params, method, param_dict):
    """returns parameters needed for method"""
    try:
        return (
            {p: params[p] for p in param_dict[method]} if method in param_dict else {}
        )
    except KeyError as k_err:
        logging.error(
            "To run method %s you need to specify parameter %s=",
            method,
            k_err.args[0],
        )
        sys.exit(1)


def _sampling_params(local_vars, bin_method: str, num_method: str = None):
    samp_params = {}
    param_dict = {
        "random": ["sampleSize", "seed"],
        "hypersampling": ["precision"],
        "onefactoratatime": ["distinctValuesPerOption"],
        "plackettburman": ["measurements", "level"],
        "kexchange": ["sampleSize", "k"],
        "twise": ["t"],
        "distance-based": ["optionWeight", "numConfigs"],
    }
    if num_method and num_method in param_dict:
        samp_params.update(
            {param: local_vars[param.lower()] for param in param_dict[num_method]}
        )
    if bin_method in param_dict:
        samp_params.update(
            {param: local_vars[param.lower()] for param in param_dict[bin_method]}
        )

    return samp_params


def _split_dataset_by_samples(data, samples):
    data = data.merge(samples, on=list(samples.columns), how="left", indicator=True)
    train = data[data["_merge"] == "both"].drop("_merge", axis=1)
    test = data[data["_merge"] == "left_only"].drop("_merge", axis=1)
    return train, test


@click.command(help="Sample using SPLConqueror.")
@click.option("--system_run_id", required=True)
@click.option("--binary_method", required=True)
@click.option("--numeric_method", default=None)
@click.option("--logs_to_artifact", type=bool, default=False)
@click.option("--samplesize", default=None)
@click.option("--seed", default=None)
@click.option("--precision", default=None)
@click.option("--distinctvaluesperoption", default=None)
@click.option("--measurements", default=None)
@click.option("--k", default=None)
@click.option("--t", default=None)
@click.option("--level", default=None)
@click.option("--optionWeight", default=None)
@click.option("--numConfigs", default=None)
def sample(
    system_run_id: str,
    binary_method: str,
    samplesize: int,
    seed: int,
    level: int,
    precision: int,
    distinctvaluesperoption: int,
    measurements: int,
    k: int,
    t: int,
    optionweight: int,
    numconfigs: int,
    numeric_method: str = None,
    logs_to_artifact: bool = False,
):
    """
    Samples valid configurations from a variability model.

    Parameters
    ----------
    n: int
        number of samples
    method: str
        method for sampling
    system_run_id : str
        run of system loading
    """
    activate_logging(logs_to_artifact)

    logging.info("Start sampling from configuration space using SPLC2py.")
    numeric_method = None if numeric_method == "None" else numeric_method
    system_cache = CacheHandler(system_run_id, new_run=False)
    feature_model = system_cache.retrieve("fm.xml")

    data = system_cache.retrieve("measurements.tsv")

    params = _sampling_params(locals(), binary_method, numeric_method)
    sampler = Sampler(feature_model, backend="local")

    with mlflow.start_run() as run:
        try:
            sampling_cache = CacheHandler(run.info.run_id)

            if not numeric_method:
                samples = pd.DataFrame(
                    sampler.sample(binary_method, formatting="dict", params=params)
                )

                train, test = _split_dataset_by_samples(data, samples)

            else:
                samples = pd.DataFrame(
                    sampler.sample(
                        binary_method, numeric_method, formatting="dict", params=params
                    )
                )

                train, test = _split_dataset_by_samples(data, samples)

            mlflow.log_param("n_train", len(train))
            mlflow.log_param("n_sampled", len(samples))
            logging.info("Save sampled configurations to cache")
            sampling_cache.save(
                {
                    "train.tsv": train,
                    "test.tsv": test,
                }
            )
            logging.info("Sampling cache dir: %s", sampling_cache.cache_dir)
            mlflow.log_artifacts(sampling_cache.cache_dir, "")
        except Exception as e:
            logging.error("During sampling the following error occured: %s", e)

        finally:
            if logs_to_artifact:
                mlflow.log_artifact("logs.txt", "")
            mlflow.log_artifacts(sampler.artifact_repo, "splc-logs")


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    sample()
