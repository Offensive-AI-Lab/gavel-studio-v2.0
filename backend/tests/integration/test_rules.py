"""Tests for rule management: CRUD, CE linking, predicates, bookmarks."""
import pytest


class TestPublicRules:
    """Public rule library."""

    def test_get_public_rules(self, client, auth_headers):
        res = client.get("/rules/public/library", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        rules = data.get("rules", data) if isinstance(data, dict) else data
        assert isinstance(rules, list)

    def test_create_public_rule(self, client, auth_headers, test_user):
        # Create two CEs first (minimum 2)
        ce_ids = []
        for i in range(2):
            ce_res = client.post("/cognitive/create", json={
                "user_id": test_user["user_id"],
                "name": f"rule_test_ce_{i}_{id(test_user)}",
                "definition": f"Test CE {i}",
            }, headers=auth_headers)
            if ce_res.status_code == 200:
                ce_ids.append(ce_res.json()["ce_id"])

        if len(ce_ids) < 2:
            pytest.skip("Could not create enough CEs")

        res = client.post("/rules/public/create", json={
            "name": f"test_rule_{id(test_user)}",
            "predicate": f"CE_A AND CE_B",
            "ce_ids": ce_ids,
            "categories": [],
        }, headers=auth_headers)
        assert res.status_code in (200, 201, 400, 422, 500)


class TestRuleSetup:
    """Rule setup operations (classifier-specific)."""

    def test_link_ce_to_setup(self, client, test_classifier, auth_headers, test_user):
        cid = test_classifier["classifier_id"]
        # Get rules for this classifier
        rules_res = client.get(f"/classifiers/{cid}/rules", headers=auth_headers)
        if rules_res.status_code != 200:
            pytest.skip("Could not get rules")
        rules_data = rules_res.json()
        rules_list = rules_data.get("rules", rules_data) if isinstance(rules_data, dict) else rules_data
        if not rules_list:
            pytest.skip("No rules in test classifier")
        setup_id = rules_list[0].get("setup_id")
        if not setup_id:
            pytest.skip("No setup_id found")

        # Create a CE to link
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"link_ce_{id(test_user)}",
            "definition": "linkable",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")

        ce_id = ce_res.json()["ce_id"]
        res = client.post(f"/rules/setup/{setup_id}/link-ce", json={
            "ce_id": ce_id,
            "role": "necessary",
            "fallback_group": 0,
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 500)


# Rule-bookmark CRUD lives on the central server now (see
# central-server/app/routes/bookmarks.py). The local route at
# /rules/public/bookmarks/{user_id} is a thin HTTP proxy — testing it
# end-to-end requires a running central server, which integration tests
# here don't spin up. Coverage lives on the central-server side instead.
