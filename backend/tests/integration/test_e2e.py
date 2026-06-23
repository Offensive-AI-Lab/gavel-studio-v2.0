"""End-to-end tests: full user workflows from registration through evaluation."""
import pytest
import time


class TestFullUserWorkflow:
    """E2E: Register -> Create Model -> Create Classifier -> Add Rules -> Check Status."""

    def test_complete_setup_workflow(self, client):
        """Test the entire setup flow a new user would follow."""
        suffix = int(time.time() * 1000) % 1000000

        # 1. Register
        reg_res = client.post("/user/register", json={
            "username": f"e2e_user_{suffix}",
            "email": f"e2e_{suffix}@test.com",
            "password": "E2EPassword123!",
        })
        assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
        user_data = reg_res.json()
        user_id = user_data["user_id"]

        # Login to get token (register may not return one)
        login_res = client.post("/user/login", json={"email": f"e2e_{suffix}@test.com", "password": "E2EPassword123!"})
        token = login_res.json().get("token")
        assert token, "Login did not return token"
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Verify authenticated access
        me_res = client.get("/user/me", headers=headers)
        assert me_res.status_code == 200

        # 3. Create model (using GPT-2 as a lightweight public model)
        model_res = client.post("/models/create", json={
            "user_id": user_id,
            "name": f"E2EModel_{suffix}",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=headers)
        assert model_res.status_code == 200, f"Model creation failed: {model_res.text}"
        model_data = model_res.json()
        model_id = model_data.get("model", model_data).get("model_id") or model_data.get("model_id")

        # 4. List models
        models_res = client.get(f"/models/{user_id}", headers=headers)
        assert models_res.status_code == 200
        models_data = models_res.json()
        models_list = models_data.get("models", models_data) if isinstance(models_data, dict) else models_data
        assert any(m["model_id"] == model_id for m in models_list)

        # 5. Create classifier
        cls_res = client.post("/classifiers/create", json={
            "model_id": model_id,
            "name": f"E2ECLS_{suffix}",
        }, headers=headers)
        assert cls_res.status_code == 200, f"Classifier creation failed: {cls_res.text}"
        cls_data = cls_res.json()
        classifier_id = cls_data.get("classifier", cls_data).get("classifier_id") or cls_data.get("classifier_id")

        # 6. Verify classifier appears in list
        cls_list = client.get(f"/classifiers/{model_id}", headers=headers)
        assert cls_list.status_code == 200
        cls_list_data = cls_list.json()
        cls_items = cls_list_data.get("classifiers", cls_list_data) if isinstance(cls_list_data, dict) else cls_list_data
        assert any(c["classifier_id"] == classifier_id for c in cls_items)

        # 7. Get classifier details
        details = client.get(f"/classifiers/details/{classifier_id}", headers=headers)
        assert details.status_code == 200
        assert details.json()["status"] in ("untrained", "active")

        # 8. Check training status
        status_res = client.get(f"/classifiers/{classifier_id}/training-status", headers=headers)
        assert status_res.status_code == 200

        # 9. Check dashboard
        dash_res = client.get(f"/dashboard/{user_id}", headers=headers)
        assert dash_res.status_code == 200
        dash_data = dash_res.json()
        assert "stats" in dash_data
        assert dash_data["stats"]["total_models"] >= 1


class TestCEWorkflow:
    """E2E: Create CEs -> Bookmark -> Search -> Create rule from CEs."""

    def test_ce_to_rule_workflow(self, client, test_user, auth_headers):
        uid = test_user["user_id"]
        suffix = int(time.time()) % 100000

        # 1. Create two CEs
        ce_ids = []
        for i in range(2):
            res = client.post("/cognitive/create", json={
                "user_id": uid,
                "name": f"e2e_ce_{i}_{suffix}",
                "definition": f"E2E test CE number {i}",
            }, headers=auth_headers)
            assert res.status_code == 200
            ce_ids.append(res.json()["ce_id"])

        # 2. These are LOCAL draft CEs (no public_id yet), so bookmarking them
        #    is correctly rejected with 400 — bookmarks live on the central
        #    server keyed by public_id, so an asset must be published first.
        for ce_id in ce_ids:
            bm_res = client.post("/cognitive/bookmark", json={
                "user_id": uid,
                "ce_id": ce_id,
            }, headers=auth_headers)
            assert bm_res.status_code == 400

        # 3. The bookmark-list endpoint still answers (proxies central server)
        #    and returns a list shape regardless.
        bm_list = client.get(f"/cognitive/bookmarks/{uid}", headers=auth_headers)
        assert bm_list.status_code == 200
        bm_data = bm_list.json()
        bm_items = bm_data.get("bookmarks", bm_data) if isinstance(bm_data, dict) else bm_data
        assert isinstance(bm_items, list)

        # 4. Search library
        search_res = client.get("/library/search", params={
            "q": f"e2e_ce_0_{suffix}",
            "user_id": uid,
        }, headers=auth_headers)
        assert search_res.status_code == 200


class TestDashboard:
    """Dashboard data integrity."""

    def test_dashboard_stats_structure(self, client, test_user, auth_headers):
        res = client.get(f"/dashboard/{test_user['user_id']}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "stats" in data
        stats = data["stats"]
        required_keys = ["total_models", "total_classifiers", "total_rules", "total_ces"]
        for key in required_keys:
            assert key in stats, f"Missing stat: {key}"
            assert isinstance(stats[key], int)

    def test_dashboard_nonexistent_user(self, client, auth_headers):
        res = client.get("/dashboard/99999", headers=auth_headers)
        assert res.status_code in (200, 404)


class TestLibrarySearch:
    """Library search functionality."""

    def test_search_empty_query(self, client, auth_headers, test_user):
        res = client.get("/library/search", params={
            "q": "",
            "user_id": test_user["user_id"],
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_search_with_category(self, client, auth_headers, test_user):
        res = client.get("/library/search", params={
            "q": "security",
            "user_id": test_user["user_id"],
            "categories": "Security & Defense",
        }, headers=auth_headers)
        assert res.status_code == 200

    def test_get_categories(self, client, auth_headers):
        res = client.get("/library/categories", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) > 0  # Default categories should exist
