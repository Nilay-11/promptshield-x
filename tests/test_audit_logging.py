import sqlite3
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.init_db import DB_PATH, SCHEMA, log_audit

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """
    Use a temporary or custom database path for testing to avoid overwriting production logs.
    """
    test_db_path = DB_PATH.parent / "test_promptshield.db"
    
    # Override the DB_PATH in init_db
    monkeypatch.setattr("app.core.init_db.DB_PATH", test_db_path)
    
    # Re-initialize test db
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except PermissionError:
            pass
        
    conn = sqlite3.connect(test_db_path)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()
    
    yield test_db_path
    
    # Cleanup test db
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except PermissionError:
            pass


def test_log_audit_direct_write(setup_test_db):
    db_path = setup_test_db
    log_audit(
        user_id="test_user_123",
        prompt="Test direct prompt injection test",
        risk_score=85,
        attack_category="jailbreak",
        action_taken="BLOCK",
        detection_evidence="direct test matches"
    )
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_log")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 1
    assert rows[0][2] == "test_user_123"
    assert rows[0][3] == "Test direct prompt injection test"
    assert rows[0][4] == 85
    assert rows[0][5] == "jailbreak"
    assert rows[0][6] == "BLOCK"
    assert rows[0][7] == "direct test matches"


def test_analyze_endpoint_logs_audit(setup_test_db, monkeypatch):
    """
    Stub the pattern scanner and semantic classifier to avoid slow inference/downloads,
    then test that /analyze writes to the test db.
    """
    def dummy_scan(text):
        return {"severity": 80, "matches": [{"id": "rule_1", "category": "jailbreak", "severity": 80}]}
        
    def dummy_classify(text):
        return {"category": "jailbreak", "confidence": 0.9, "raw_scores": {}}
        
    monkeypatch.setattr("app.api.routes.scan_prompt", dummy_scan)
    monkeypatch.setattr("app.api.routes.classify_prompt", dummy_classify)
    
    response = client.post("/analyze", json={
        "prompt": "Test analyze endpoint logging prompt",
        "user_id": "api_user_456"
    })
    
    assert response.status_code == 200
    
    # Now verify database contains this entry
    db_path = setup_test_db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, prompt, risk_score, action_taken FROM audit_log")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 1
    assert rows[0][0] == "api_user_456"
    assert rows[0][1] == "Test analyze endpoint logging prompt"
    assert rows[0][3] == "BLOCK"


def test_admin_logs_endpoint(setup_test_db):
    # Add dummy logs
    log_audit("user1", "prompt1", 10, "safe", "PASS", "evidence1")
    log_audit("user2", "prompt2", 90, "jailbreak", "BLOCK", "evidence2")
    
    response = client.get("/admin/logs?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["logs"]) == 2
    
    # Newest log first
    assert body["logs"][0]["user_id"] == "user2"
    assert body["logs"][0]["prompt"] == "prompt2"
    assert body["logs"][1]["user_id"] == "user1"


def test_dashboard_endpoint_serves_html():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PromptShield X" in response.text


def test_root_redirects_to_dashboard():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app"  # root opens the user gateway app

