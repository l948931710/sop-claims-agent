"""The two pure-code safety layers: identity matching and the output guards."""

from __future__ import annotations

import pytest

from app.guards import check_reply
from app.identity import canonical, evaluate

# -- identity -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("full_name", "  margaret   chen ", "margaret chen"),
        ("dob", "1985-03-15", "1985-03-15"),
        ("dob", "03/15/1985", "1985-03-15"),
        ("dob", "March 15th, 1985", "1985-03-15"),
        ("dob", "the fifteenth", None),
        ("phone", "+1 (650) 521-2836", "6505212836"),
        ("phone", "521-2836", None),
        ("email", "Margaret@Email.com.", "margaret@email.com"),
        ("email", "not-an-email", None),
        ("id_last4", "4472", "4472"),
        ("id_last4", "123-45-4472", "4472"),  # a full SSN is cut down to its last four
        ("id_last4", "44", None),
        ("policy_number", "pol 9921", "POL-9921"),
    ],
)
def test_canonical_normalises_what_callers_say(field, raw, expected):
    assert canonical(field, raw) == expected


def test_three_matching_fields_verify(repo):
    v = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}, repo.policyholders())
    assert v.verified and v.party_id == "P9"
    assert sorted(v.matched) == ["dob", "full_name", "id_last4"]


def test_two_matching_fields_do_not_verify(repo):
    v = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15"}, repo.policyholders())
    assert not v.verified and v.record_located
    assert v.missing == ["phone", "email", "id_last4"]


def test_policy_number_locates_the_record_but_does_not_count(repo):
    v = evaluate({"policy_number": "POL-9921", "full_name": "Margaret Chen", "dob": "1985-03-15"}, repo.policyholders())
    assert v.policy_number_matched is True
    assert len(v.matched) == 2 and not v.verified


def test_alternate_fields_work_as_well_as_the_ssn(repo):
    v = evaluate({"full_name": "Chen, Margaret", "phone": "6505212836", "email": "margaret@email.com"}, repo.policyholders())
    assert v.verified and v.party_id == "P9"


def test_a_wrong_value_is_reported_as_a_mismatch_not_a_match(repo):
    v = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-16", "id_last4": "4472"}, repo.policyholders())
    assert v.mismatched == ["dob"] and not v.verified


def test_aliases_and_national_id_holders_verify(repo):
    v = evaluate({"full_name": "Yaven Li", "email": "yawen.li@example.com", "id_last4": "5317"}, repo.policyholders())
    assert v.verified and v.party_id == "P13"
    v = evaluate({"full_name": "Ma Tian", "dob": "1964-09-10", "id_last4": "6688"}, repo.policyholders())
    assert v.verified and v.party_id == "P12"


def test_details_from_two_different_people_do_not_pick_a_record(repo):
    v = evaluate({"full_name": "Margaret Chen", "dob": "1990-08-21"}, repo.policyholders())
    assert v.ambiguous and not v.record_located and not v.verified


def test_unknown_caller_matches_nothing(repo):
    v = evaluate({"full_name": "Nobody Here", "dob": "2000-01-01"}, repo.policyholders())
    assert not v.record_located and sorted(v.mismatched) == ["dob", "full_name"]


# -- guards -------------------------------------------------------------------

CASE_FACTS = (
    '{"case_id": "CL-2048", "filed_on": "2026-01-12", "appeal_deadline": "2026-03-18", '
    '"allowed_max_amount": "1450.00", "net_pay": "0.00", "net_fee": "1450.00"}'
)


def guard(reply, repo, *corpus):
    return check_reply(reply, corpus=list(corpus), policyholders=repo.policyholders())


def test_grounded_reply_passes_in_any_wording(repo):
    reply = "Your claim CL-2048 was filed on January 12, 2026. The allowed maximum is $1,450 and the appeal deadline is March 18."
    assert guard(reply, repo, CASE_FACTS) == []


def test_claim_number_not_in_the_facts_is_rejected(repo):
    assert any("CL-2102" in v for v in guard("I also see claim CL-2102.", repo, CASE_FACTS))


def test_invented_amount_is_rejected(repo):
    assert any("$900" in v for v in guard("You'll receive $900.", repo, CASE_FACTS))


def test_difference_of_two_grounded_amounts_is_allowed(repo):
    facts = '{"allowed_max_amount": "800.00", "net_pay": "780.00"}'
    assert guard("That leaves $20 between the allowed maximum of $800 and the $780 paid.", repo, facts) == []


def test_invented_date_is_rejected(repo):
    violations = guard("Please send them by March 25, 2026.", repo, CASE_FACTS)
    assert any("March 25, 2026" in v for v in violations)


def test_month_and_year_alone_are_checked_too(repo):
    assert guard("It was filed in January 2026.", repo, CASE_FACTS) == []
    assert guard("It was filed in February 2026.", repo, CASE_FACTS) != []


def test_identifier_on_file_cannot_appear_unless_the_caller_gave_it(repo):
    leak = "I have your number as 650-521-2836."
    assert any("phone number on file" in v for v in guard(leak, repo, "My name is Margaret Chen"))
    assert guard(leak, repo, "My phone is (650) 521-2836") == []


def test_id_digits_and_birth_date_on_file_are_caught(repo):
    assert guard("The last four we have are 4472.", repo, "hello") != []
    assert guard("Your birthday is March 15, 1985.", repo, "hello") != []
