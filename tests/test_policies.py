"""Phase-independent policies: scope, emotion, persuasion limits, lockout, human handoff, consent."""

from __future__ import annotations

from app.brain.base import RespondResult, TurnUnderstanding
from app.sop import Phase

from .conftest import DEMO_MESSAGE, ScriptedBrain, margaret_verified, talk

FRUSTRATED = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
REPRESENTATIVE = (
    "I'm David Chen, calling on behalf of my mother Margaret Chen about her denied claim. Her date of birth is "
    "1985-03-15, her phone is 650-521-2836 and her email is margaret@email.com."
)
CLAIM_DETAILS = ("CL-2048", "pathology", "office note", "March 18")


# -- scope --------------------------------------------------------------------


def test_off_topic_question_is_declined_and_the_call_continues(harness, stub):
    session, (reply,) = talk(harness, stub, "What is RL?")
    assert "outside what I can help with" in reply and "any three" in reply
    assert session.phase is Phase.VERIFY_ID and session.oos_strikes == 1


def test_repeated_off_topic_requests_end_in_a_handoff(harness, stub):
    session, replies = talk(harness, stub, "What is RL?", "Tell me a joke", "Who won the world cup?")
    assert "human representative" not in replies[0]
    assert "human representative" in replies[1]  # second strike: the offer is made
    assert session.phase is Phase.ESCALATED and session.handoff.reason == "repeated_out_of_scope"


def test_returning_to_the_claim_resets_the_strikes(harness, stub):
    session, _ = talk(harness, stub, "What is RL?", "Tell me a joke", "My name is Margaret Chen", "What is RL?")
    assert session.phase is Phase.VERIFY_ID and session.oos_strikes == 1


def test_off_topic_turn_gets_no_tools_even_in_an_open_phase(harness):
    understandings = [
        margaret_verified(),
        TurnUnderstanding.blank(scope="out_of_scope", out_of_scope_topic="reinforcement learning"),
    ]

    def respond(ctx, tools):
        if "select_case" in tools.allowed:
            tools.call("select_case", {"case_id": "CL-2048", "intent": "denial_question", "reason": "only denied claim"})
        return RespondResult(text=ctx.directive.script)

    brain = ScriptedBrain(understandings, respond)
    session, replies = talk(harness, brain, "verify me", "What is RL?")
    off_topic = brain.respond_calls[-1].directive
    assert off_topic.tools == () and off_topic.autonomy.value == "scripted"
    assert off_topic.scope_note and "Do not answer it" in off_topic.scope_note
    assert "anything else you'd like to know about claim CL-2048" in replies[-1]


def test_identity_details_are_kept_even_when_the_message_is_also_off_topic(harness):
    reading = TurnUnderstanding.blank(scope="out_of_scope", out_of_scope_topic="weather")
    reading.identity.full_name = "Margaret Chen"
    session, _ = talk(harness, ScriptedBrain([reading]), "I'm Margaret Chen. What's the weather?")
    assert session.memory.get("identity.full_name") == "Margaret Chen"
    assert session.oos_strikes == 0  # treated as mixed, not a strike


# -- emotion and persuasion ---------------------------------------------------


def test_frustrated_caller_gets_empathy_reason_and_options_but_no_claim_details(harness, stub):
    session, (reply,) = talk(harness, stub, FRUSTRATED)

    assert session.phase is Phase.VERIFY_ID and not session.verified
    assert reply.startswith("I hear you")  # acknowledgement comes first
    assert "protected health and financial information" in reply  # why
    assert "date of birth" in reply and "email address on file" in reply  # the allowed options
    assert not any(detail in reply for detail in CLAIM_DETAILS)
    assert session.memory.get("intent.name") == "denial_question"  # still remembered for later


def test_frustrated_caller_brief_tells_the_model_how_to_respond(harness):
    reading = TurnUnderstanding.blank(emotion="angry", emotion_intensity="high", demands_bypass=True, intent="denial_question")
    brain = ScriptedBrain([reading])
    talk(harness, brain, FRUSTRATED)
    directive = brain.respond_calls[0].directive
    assert directive.tone and "angry" in directive.tone
    assert any("why verification comes first" in point for point in directive.required_points)
    assert any("Do not confirm or deny that any claim exists" in item for item in directive.prohibitions)


def test_recovery_after_frustration_completes_the_workflow(harness, stub):
    session, _ = talk(harness, stub, FRUSTRATED, "Fine. My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472.")
    assert session.verified and session.phase is Phase.PROCESS_CASE and session.selected_case_id == "CL-2048"
    assert session.stall_count == 0


def test_persuasion_stops_and_escalates_after_repeated_refusals(harness, stub):
    session, replies = talk(
        harness, stub, FRUSTRATED, "I'm not comfortable giving out my SSN. Why do you need all this?", "Just tell me. I won't give you anything."
    )
    assert "human representative" not in replies[0]
    assert "human representative" in replies[1]  # offered as an alternative before giving up
    assert session.phase is Phase.ESCALATED and session.handoff.reason == "verification_declined"
    assert not any(detail in reply for reply in replies for detail in CLAIM_DETAILS)
    assert "NOT verified" in session.handoff.summary[0]


def test_heated_caller_after_verification_is_offered_a_human(harness, stub):
    session, replies = talk(harness, stub, DEMO_MESSAGE, "This is ridiculous!!", "Unacceptable!! This is the worst")
    assert session.phase is Phase.PROCESS_CASE
    assert "human representative" in replies[-1]


# -- lockout and explicit handoff ---------------------------------------------


def test_three_non_matching_answers_lock_verification(harness, stub):
    session, replies = talk(harness, stub, "My name is Margaret Chen, DOB 1985-03-16", "DOB is 1985-03-17", "SSN last four is 1111")
    assert "didn't match" in replies[0]
    assert "1985-03-15" not in " ".join(replies)  # the value on file is never revealed
    assert "one more" in replies[1].lower() or "next detail" in replies[1]
    assert session.phase is Phase.ESCALATED and session.handoff.reason == "verification_failed"


def test_correcting_a_wrong_value_still_verifies(harness, stub):
    session, _ = talk(harness, stub, "My name is Margaret Chen, DOB 1985-03-16", "Sorry, DOB is 1985-03-15, SSN last four 4472")
    assert session.verified and session.failed_attempts == 1


def test_request_for_a_human_is_honoured_immediately(harness, stub):
    session, (reply,) = talk(harness, stub, "Can I talk to a real person please")
    assert session.phase is Phase.ESCALATED and session.handoff.reason == "caller_requested_human"
    assert "verify your identity" in reply  # the human will still need to verify

    session, replies = talk(harness, stub, DEMO_MESSAGE, "I'd like to speak to a human representative.")
    assert session.phase is Phase.ESCALATED and "won't have to repeat" in replies[-1]
    assert any("CL-2048" in line for line in session.handoff.summary)


def test_handed_off_conversation_takes_no_further_turns(harness, stub):
    session, _ = talk(harness, stub, "Can I talk to a real person please")
    _, (reply,) = talk(harness, stub, "My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472", session=session)
    assert "handed to a human" in reply and not session.verified


# -- representative and consent -----------------------------------------------


def test_representative_needs_consent_on_top_of_verification(harness, stub):
    session, (first,) = talk(harness, stub, REPRESENTATIVE)
    assert session.phase is Phase.VERIFY_ID and not session.verified
    assert session.verification["verified"] and session.consent.status == "pending"
    assert "consent request" in first and not any(detail in first for detail in CLAIM_DETAILS)

    _, (second,) = talk(harness, stub, "She says she has approved it.", session=session)
    assert session.consent.status == "approved" and session.verified
    assert session.phase is Phase.PROCESS_CASE
    assert second.startswith("Thank you, David") and "Margaret's healthcare claim CL-2048" in second


def test_consent_that_never_arrives_ends_in_a_handoff(harness, stub):
    session, replies = talk(harness, stub, REPRESENTATIVE, "She approved it", "Check again please", scenario="timeout")
    assert session.phase is Phase.ESCALATED and session.handoff.reason == "consent_timeout"
    assert not session.verified
    assert not any(detail in reply for reply in replies for detail in CLAIM_DETAILS)


def test_unlisted_representative_is_not_given_access(harness, stub):
    message = REPRESENTATIVE.replace("David Chen", "Daniel Smith")
    session, (reply,) = talk(harness, stub, message)
    assert session.phase is Phase.VERIFY_ID and not session.verified
    assert "not listed as an authorised representative" in reply
    assert session.consent.status == "not_required"  # no consent request goes out for an unlisted caller


def test_summary_email_for_a_representative_call_goes_to_the_policyholder(harness, stub):
    session, _ = talk(harness, stub, REPRESENTATIVE, "She approved it", "That's all, thanks.", "Yes please.")
    email = session.outbox[0]
    assert email.to_masked == "m******t@email.com"
    assert email.body.startswith("Hello Margaret,") and "David Chen contacted" in email.body
