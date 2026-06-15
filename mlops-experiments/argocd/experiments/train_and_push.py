import os
import shutil
import joblib
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")

EXPERIMENT_NAME = "Iris Classification"


def push_metrics(run_id: str, accuracy: float, loss: float):
    registry = CollectorRegistry()

    accuracy_gauge = Gauge(
        "mlflow_accuracy",
        "Accuracy of MLflow experiment",
        ["run_id"],
        registry=registry,
    )

    loss_gauge = Gauge(
        "mlflow_loss",
        "Loss of MLflow experiment",
        ["run_id"],
        registry=registry,
    )

    accuracy_gauge.labels(run_id=run_id).set(accuracy)
    loss_gauge.labels(run_id=run_id).set(loss)

    push_to_gateway(
        PUSHGATEWAY_URL,
        job="mlflow_experiment",
        registry=registry,
    )


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    learning_rates = [0.001, 0.01, 0.05]
    epochs_list = [100, 200, 400]

    os.makedirs("models", exist_ok=True)
    os.makedirs("../best_model", exist_ok=True)

    best_accuracy = -1
    best_model_path = None
    best_run_id = None

    for learning_rate in learning_rates:
        for epochs in epochs_list:
            with mlflow.start_run() as run:
                run_id = run.info.run_id

                model = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    learning_rate_init=learning_rate,
                    max_iter=epochs,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                loss = log_loss(y_test, y_proba)

                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("epochs", epochs)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("loss", loss)

                local_model_path = f"models/model_{run_id}.joblib"
                joblib.dump(model, local_model_path)

                mlflow.log_artifact(local_model_path)
                mlflow.sklearn.log_model(model, "model")

                push_metrics(run_id, accuracy, loss)

                print(
                    f"run_id={run_id} | "
                    f"learning_rate={learning_rate} | "
                    f"epochs={epochs} | "
                    f"accuracy={accuracy:.4f} | "
                    f"loss={loss:.4f}"
                )

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = local_model_path
                    best_run_id = run_id

    shutil.copy(best_model_path, "../best_model/model.joblib")

    print("\n✅ Best model selected")
    print(f"run_id={best_run_id}")
    print(f"accuracy={best_accuracy:.4f}")
    print("saved_to=../best_model/model.joblib")


if __name__ == "__main__":
    main()
