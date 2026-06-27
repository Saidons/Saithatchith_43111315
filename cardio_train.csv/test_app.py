from app import app


def assert_ok(response, endpoint):
    if response.status_code != 200:
        raise AssertionError(f"{endpoint} failed: {response.status_code} {response.get_data(as_text=True)}")
    payload = response.get_json()
    if not payload:
        raise AssertionError(f"{endpoint} returned no JSON payload")
    return payload


def main():
    client = app.test_client()

    health = assert_ok(client.get("/api/health"), "/api/health")
    print(f"Health: {health['status']}")

    initialized = assert_ok(client.post("/api/initialize"), "/api/initialize")
    print(
        "Initialized:",
        initialized["dataset"]["source"],
        f"({initialized['dataset']['records']} records)",
    )
    print("Metrics:", ", ".join(initialized["metrics"].keys()))

    values = {feature["name"]: feature["value"] for feature in initialized["features"]}
    prediction = assert_ok(client.post("/api/predict", json={"values": values}), "/api/predict")
    print(
        "Prediction:",
        prediction["risk_class"],
        f"{prediction['risk_percentage']:.1f}%",
    )

    importance = assert_ok(client.get("/api/feature-importance"), "/api/feature-importance")
    if not importance["image"].startswith("data:image/png;base64,"):
        raise AssertionError("Feature importance image was not returned as base64 PNG")
    print("Feature importance chart: ok")

    confusion = assert_ok(client.get("/api/confusion-matrix"), "/api/confusion-matrix")
    if not confusion["image"].startswith("data:image/png;base64,"):
        raise AssertionError("Confusion matrix image was not returned as base64 PNG")
    print("Confusion matrix chart: ok")

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
