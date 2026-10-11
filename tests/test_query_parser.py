import pytest

from app.agent.query_parser import parse_query


@pytest.mark.parametrize("q,expected", [
    ("cowboy boots", {"category": "western"}),
    ("waterproof hiking boots under $200", {"category": "hiking", "waterproof": True, "max_price": 200}),
    ("steel toe boots", {"safety_toe": "steel"}),
    ("safety boots over 150", {"safety_toe": "any", "min_price": 150}),
    ("soft toe work boots", {"category": "work", "safety_toe": "none"}),
    ("not waterproof", {"waterproof": False}),
    ("burgundy chelsea", {"category": "chelsea", "color": "oxblood"}),
])
def test_filters(q, expected):
    f = parse_query(q)["filters"].model_dump(exclude_none=True)
    assert f == expected


def test_intent_and_clarification():
    assert parse_query("BOOT-001")["intent"] == "keyword_lookup"
    assert parse_query("boots")["needs_clarification"] is True
    assert parse_query("hiking boots")["needs_clarification"] is False
