from kfp import compiler
from kfp.dsl import pipeline

from components import load_data, train_model


@pipeline(
    name="Simple ML Pipeline",
    description="Simple ML Pipeline using Kubeflow"
)
def ml_pipeline():

    data = load_data()

    train_model(
      X=data.outputs["X"],
      y=data.outputs["y"]
)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="ml_pipeline.yaml"
    )

    print("Pipeline compiled successfully!")