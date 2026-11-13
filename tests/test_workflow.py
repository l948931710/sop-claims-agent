"""The workflow end to end, driven through the offline stub (no model)."""

from __future__ import annotations

from app.sop import Phase

from .conftest import DEMO_MESSAGE, VERIFY_MARGARET, talk

CLAIM_DETAILS = ("CL-2048", "pathology", "office note", "March 18", "1,450", "denied because")


def test_demo_case_verifies_remembers_and_resolves_in_one_turn(harness, stub):
    session, (reply,) = talk(harness, stub, DEMO_MESSAGE)

    assert session.verified and session.party_id == "P9"
    assert session.phase is Phase.PROCESS_CASE
    assert session.selected_case_id == "CL-2048" and session.selected_intent == "denial_question"
    assert "CL-2048" in reply and "pathology report" in reply

    # The hint was heard during VERIFY_ID, kept, and used after verification.
    hint = session.memory.slots["case.description"]
    assert hint.phase == "VERIFY_ID" and "denied healthcare claim from January" in hint.value
    assert hint.used
    transitions = [e.title for e in session.traces[-1].events if e.kind == "transition"]
    assert transitions == ["VERIFY_ID → RESOLVE_INTENT", "RESOLVE_INTENT → PROCESS_CASE"]


def test_reason_given_before_verification_is_stored_but_not_acted_on(harness, stub):
    session, (first,) = talk(harness, stub, "Hi, this is Margaret Chen. I'm calling about my denied healthcare claim from January.")

    assert session.phase is Phase.VERIFY_ID and not session.verified
    assert session.memory.get("case.status") == "denied"
    assert session.memory.get("intent.name") == "denial_question"
    assert not any(detail in first for detail in CLAIM_DETAILS)
    assert "verified" in first  # "we'll get to it as soon as you're verified"

    # Verification completes on the next turn and the remembered hint resolves the case: no "which claim?".
    _, (second,) = talk(harness, stub, "My date of birth is 1985-03-15 and the last four of my SSN are 4472.", session=session)
    assert session.phase is Phase.PROCESS_CASE and session.selected_case_id == "CL-2048"
    assert "Which one" not in second and "CL-2048" in second


def test_partial_answers_accumulate_across_turns(harness, stub):
    session, replies = talk(harness, stub, "My name is Margaret Chen", "My email is margaret@email.com")
    assert session.phase is Phase.VERIFY_ID
    assert sorted(session.verification["matched"]) == ["email", "full_name"]
    assert "1 more" in replies[-1]
    talk(harness, stub, "And my phone is 650-521-2836", session=session)
    assert session.verified


def test_ambiguous_hint_leads_to_a_question_not_a_guess(harness, stub):
    session, (reply,) = talk(harness, stub, VERIFY_MARGARET + " I have a question about my healthcare claim.")
    assert session.phase is Phase.RESOLVE_INTENT and session.selected_case_id is None
    assert "CL-2048" in reply and "CL-2011" in reply and "CL-2102" not in reply  # only the two healthcare claims


def test_caller_can_move_to_a_different_claim(harness, stub):
    session, _ = talk(harness, stub, DEMO_MESSAGE, "What about my auto claim?")
    assert session.phase is Phase.PROCESS_CASE and session.selected_case_id == "CL-2102"
    assert [log.case_id for log in session.cases] == ["CL-2048", "CL-2102"]


def test_follow_up_answers_come_from_the_guidance_tool(harness, stub):
    session, (_, reply) = talk(harness, stub, DEMO_MESSAGE, "How do I submit the documents?")
    tools = [e for e in session.traces[-1].events if e.kind == "tool"]
    assert [t.title for t in tools] == ["get_followup_guidance"]
    assert "member portal" in reply


def test_policyholder_with_no_claims_is_told_so(harness, stub):
    session, (reply,) = talk(harness, stub, "My name is Ava Lopez, DOB 1990-08-21, email ava.lopez@email.com")
    assert session.verified and session.phase is Phase.RESOLVE_INTENT
    assert "don't see any claims" in reply


def test_full_call_with_email_sent(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE, "How do I submit the documents?", "That's all, thanks.")
    assert session.phase is Phase.POST_PROCESS and session.email.status == "offered"
    assert "m******t@email.com" in replies[-1] and not session.outbox  # offered, nothing sent yet

    talk(harness, stub, "Yes please.", session=session)
    assert session.email.status == "sent" and len(session.outbox) == 1
    email = session.outbox[0]
    assert email.to_masked == "m******t@email.com"
    for expected in ("CL-2048", "denied", "pathology report", "March 18, 2026", "submission method"):
        assert expected in email.body

    _, (goodbye,) = talk(harness, stub, "No, bye.", session=session)
    assert session.phase is Phase.CLOSED and "goodbye" in goodbye.lower()


def test_full_call_with_email_skipped(harness, stub):
    session, _ = talk(harness, stub, DEMO_MESSAGE, "That's all, thanks.", "No, skip the email.")
    assert session.email.status == "skipped" and session.outbox == []
    talk(harness, stub, "bye", session=session)
    assert session.phase is Phase.CLOSED


def test_unclear_answer_to_the_email_offer_sends_nothing(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE, "That's all, thanks.", "hmm")
    assert session.email.status == "offered" and session.outbox == []
    assert "skip" in replies[-1]


def test_new_question_during_wrap_up_goes_back_to_the_claim(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE, "That's all, thanks.", "Actually, how do I submit the documents?")
    assert session.phase is Phase.PROCESS_CASE and session.selected_case_id == "CL-2048"
    assert "member portal" in replies[-1]
    assert session.email.status == "not_offered"  # the offer is made again at the next wrap-up


def test_email_wish_stated_early_is_remembered_and_confirmed_not_assumed(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE + " And please email me a summary when we're done.", "That's all, thanks.")
    assert session.memory.slots["prefs.email_summary"].phase == "VERIFY_ID"
    assert session.email.status == "offered" and session.outbox == []
    assert "mentioned earlier" in replies[-1]


def test_email_refusal_stated_early_is_honoured_without_asking_again(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE + " And don't email me anything.", "That's all, thanks.")
    assert session.email.status == "skipped" and session.outbox == []
    assert "mentioned earlier" in replies[-1]


def test_closed_conversation_takes_no_further_turns(harness, stub):
    session, _ = talk(harness, stub, DEMO_MESSAGE, "That's all, thanks.", "No, skip the email.", "bye")
    turns_before = session.turn
    _, (reply,) = talk(harness, stub, "Tell me about claim CL-2048 again", session=session)
    assert "ended" in reply and session.turn == turns_before


def test_goodbye_and_email_decision_in_one_message_close_the_call(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE, "No, that's all. Don't email me anything.")
    assert session.phase is Phase.CLOSED and session.email.status == "skipped" and session.outbox == []
    assert "won't send an email" in replies[-1] and "?" not in replies[-1]

    session, replies = talk(harness, stub, DEMO_MESSAGE, "That's all, thanks. Please email me the summary.")
    assert session.phase is Phase.CLOSED and session.email.status == "sent" and len(session.outbox) == 1
    assert "m******t@email.com" in replies[-1]
