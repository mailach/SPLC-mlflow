import sys
import os
from caching import CacheHandler
import pandas as pd
from rich.logging import RichHandler
import logging
import mlflow
import click

from splc2py.learning import Model


def activate_logging(logs_to_artifact):
    if logs_to_artifact:
        return logging.basicConfig(
            filename="logs.txt",
            level=logging.INFO,
            format="LEARNING    %(message)s",
        )
    return logging.basicConfig(
        level=logging.INFO,
        format="LEARNING    %(message)s",
        handlers=[RichHandler()],
    )


def _load_data(data_file: str, cache: CacheHandler, nfp: str):
    data = cache.retrieve(data_file)
    nfp = "nfp_" + nfp
    columns_to_drop = [col for col in data.columns if "nfp_" in col and col != nfp]
    data = data.drop(columns_to_drop, axis=1)
    Y = data[nfp]
    X = data.drop(nfp, axis=1)
    return X, Y


def _predict_on_test(learner, test_x: pd.DataFrame, test_y: pd.Series):
    pred = pd.Series(learner.predict(test_x))

    prediction = pd.concat([pred, test_y], axis=1)
    prediction.columns = ["predicted", "measured"]
    return prediction


def _get_params(context):
    params = [
        arg.replace("--", "")
        .replace("_", "-")
        .replace("True", "true")
        .replace("False", "false")
        for arg in context.args
    ]
    logging.error(params)
    params = {p.split("=")[0]: p.split("=")[1] for p in params}
    return params


@click.command(
    help="Learn from configurations using SPLConqueror.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--sampling_run_id", required=True)
@click.option("--nfp")
@click.option("--logs_to_artifact", type=bool, default=False)
@click.pass_context
def learn(context, sampling_run_id, nfp, logs_to_artifact):
    activate_logging(logs_to_artifact)
    ml_params = _get_params(context)
    sampling_cache = CacheHandler(sampling_run_id, new_run=False)
    train_x, train_y = _load_data("train.tsv", sampling_cache, nfp)
    train = pd.concat([train_x, train_y], axis=1)
    test_x, test_y = _load_data("test.tsv", sampling_cache, nfp)

    with mlflow.start_run() as run:
        model_cache = CacheHandler(run.info.run_id)
        logging.info("Use ml settings: %s", ml_params)
        model = Model("local")
        model.fit(train, "nfp_" + nfp, ml_params)
        logging.info("Finished training model")
        logging.info("Predict test set and save to cache.")
        prediction = _predict_on_test(model, test_x, test_y)
        model_cache.save({"predicted.tsv": prediction})
        mlflow.log_artifact(os.path.join(model_cache.cache_dir, "predicted.tsv"), "")
        if logs_to_artifact:
            mlflow.log_artifact("logs.txt", "")


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    learn()
