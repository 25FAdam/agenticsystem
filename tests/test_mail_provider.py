import pytest

from email_assistant.schemas import ReplyDraft
from email_assistant.tools.mail import MockMailProvider
from tests.conftest import INBOX, USER_EMAIL


def test_fetch_unread_excludes_read_threads(mail):
    ids = {t.id for t in mail.fetch_unread()}
    assert "t-005" not in ids  # we sent the last message, thread is read
    assert "t-001" in ids


def test_fetch_unread_is_newest_first(mail):
    threads = mail.fetch_unread()
    dates = [t.last_message.date for t in threads]
    assert dates == sorted(dates, reverse=True)


def test_get_thread_unknown_id_raises(mail):
    with pytest.raises(KeyError):
        mail.get_thread("t-999")


def test_send_reply_appends_message_and_records_sent(mail):
    before = len(mail.get_thread("t-002").messages)
    draft = ReplyDraft(
        thread_id="t-002",
        to=["dana.reyes@northwindlabs.com"],
        subject="Re: Question about API rate limits",
        body="Hi Dana, the burst raise is possible — details below.",
    )
    message = mail.send_reply(draft)

    thread = mail.get_thread("t-002")
    assert len(thread.messages) == before + 1
    assert thread.last_message.id == message.id
    assert message.sender == USER_EMAIL
    assert mail.sent == [message]


def test_apply_labels_deduplicates(mail):
    mail.apply_labels("t-002", ["client", "api", "api"])
    assert mail.get_thread("t-002").labels == ["client", "api"]


def test_archive_removes_from_unread(mail):
    mail.archive("t-004")
    thread = mail.get_thread("t-004")
    assert thread.archived and not thread.unread
    assert "t-004" not in {t.id for t in mail.fetch_unread()}


def test_mutations_do_not_touch_fixture_file(mail):
    mail.archive("t-004")
    mail.mark_read("t-001")

    fresh = MockMailProvider(INBOX, user_email=USER_EMAIL)
    assert not fresh.get_thread("t-004").archived
    assert fresh.get_thread("t-001").unread
