import io
import requests
from pathlib import Path

BASE_BACKEND = "http://127.0.0.1:8000"
BASE_FRONTEND = "http://localhost:3000"

def test_all():
    print("=" * 76)
    print("TESTING BACKEND AND FRONTEND ROUTE COMPLIANCE")
    print("=" * 76)

    # 1. GET Endpoints
    get_endpoints = [
        f"{BASE_BACKEND}/",
        f"{BASE_BACKEND}/health",
        f"{BASE_BACKEND}/api/health",
        f"{BASE_BACKEND}/api/health/",
        f"{BASE_BACKEND}/metrics",
        f"{BASE_BACKEND}/api/metrics",
        f"{BASE_BACKEND}/predict",
        f"{BASE_BACKEND}/api/predict",
        f"{BASE_BACKEND}/docs",
        f"{BASE_FRONTEND}/",
        f"{BASE_FRONTEND}/detection",
        f"{BASE_FRONTEND}/api/health",
    ]

    print("\n--- 1. Testing GET Endpoints ---")
    for u in get_endpoints:
        r = requests.get(u, allow_redirects=True, timeout=5)
        print(f"GET  {u:38} -> Status {r.status_code}")
        assert r.status_code == 200, f"Expected 200 for {u}, got {r.status_code}: {r.text}"

    # 2. POST Endpoints with Valid Dataset Image
    print("\n--- 2. Testing POST Prediction Endpoints with Dataset Image ---")
    sample_img = next(Path("DATASET/Alternaria leaf spot").glob("*.jpg"))
    print(f"Using test sample: {sample_img.name}")

    post_endpoints = [
        f"{BASE_BACKEND}/api/predict",
        f"{BASE_BACKEND}/predict",
        f"{BASE_FRONTEND}/api/predict",
    ]

    for u in post_endpoints:
        with open(sample_img, "rb") as f:
            r = requests.post(
                u,
                files={"file": (sample_img.name, f, "image/jpeg")},
                data={"model": "mobilenetv2"},
                allow_redirects=True,
                timeout=10,
            )
        print(f"POST {u:38} -> Status {r.status_code}")
        assert r.status_code == 200, f"Expected 200 for {u}, got {r.status_code}: {r.text}"
        data = r.json()
        assert data.get("status") == "success", f"Prediction status failed: {data}"
        assert "final_prediction" in data, f"Missing final_prediction: {data}"
        assert "final_label" in data, f"Missing final_label: {data}"
        print(f"     Diagnosis: {data['final_label']} ({data['confidence']*100:.1f}%), Explanation: {data['explanation']}")

    # 3. Error Handling Test (Invalid Image File)
    print("\n--- 3. Testing Error Handling on Corrupt / Non-Image File ---")
    corrupt_content = b"THIS_IS_NOT_A_VALID_IMAGE_FILE"
    r_err = requests.post(
        f"{BASE_BACKEND}/api/predict",
        files={"file": ("corrupt.jpg", io.BytesIO(corrupt_content), "image/jpeg")},
        data={"model": "mobilenetv2"},
        timeout=10,
    )
    print(f"POST {BASE_BACKEND}/api/predict (Corrupt File) -> Status {r_err.status_code}")
    assert r_err.status_code == 400, f"Expected 400 for corrupt image, got {r_err.status_code}: {r_err.text}"
    err_json = r_err.json()
    print(f"     Clean Error Message: {err_json.get('detail')}")

    print("\n" + "=" * 76)
    print("ALL ROUTE AND PREDICTION TESTS PASSED WITH 0 NOT FOUND (404) ERRORS!")
    print("=" * 76)

if __name__ == "__main__":
    test_all()
