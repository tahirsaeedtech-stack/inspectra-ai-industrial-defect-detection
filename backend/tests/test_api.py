import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_test_image():
    image = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["application"] == "Inspectra AI"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(client):
    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": ("test.png", image, "image/png")
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

    assert len(data["probabilities"]) == 6
    assert 0 <= data["confidence"] <= 1


def test_invalid_file(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_gradcam(client):
    image = create_test_image()

    response = client.post(
        "/explain",
        files={
            "file": ("test.png", image, "image/png")
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "gradcam_image" in data

    assert data["gradcam_image"].startswith(
        "data:image/png;base64,"
    )
