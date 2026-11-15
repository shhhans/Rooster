"""Error types raised by the timeline kernel.

These map directly to the error codes declared in the frozen contract:
INVALID_INPUT, UNKNOWN_TIMEZONE, EVENT_NOT_FOUND,
EVENT_ALREADY_SUPERSEDED, EVENT_RETRACTED.
"""

from __future__ import annotations


class TimelineError(Exception):
    """Base class for all timeline kernel errors."""


class InvalidInputError(TimelineError):
    """Required field missing, empty string where non-empty required,
    unknown status value, payload not a JSON object, or any other input
    validation failure."""


class UnknownTimezoneError(TimelineError):
    """Timezone is not a recognized IANA name, or the local time does
    not exist in that zone (DST spring-forward gap)."""


class EventNotFoundError(TimelineError):
    """The event_id does not exist in storage."""


class EventAlreadySupersededError(TimelineError):
    """Target event has a newer correction; error payload includes
    ``superseded_by`` so the caller can re-issue against the latest head."""

    def __init__(self, message: str, superseded_by: str):
        super().__init__(message)
        self.superseded_by = superseded_by


class EventRetractedError(TimelineError):
    """Target event is retracted; corrections to retracted events are
    not allowed — re-ingest as a fresh event with a new dedupe_key."""


class EventPayloadInvalidError(TimelineError):
    """A schedule-state-input event has a malformed payload.

    Raised by ``derive_schedule_state_inputs`` when a referenced event
    (title in {flight, duty, rest}) has a missing required payload field,
    a non-object payload, a non-string ``flight_id``, a missing-offset or
    unparseable payload timestamp, or an invalid IANA name in
    ``payload.origin_tz`` / ``payload.destination_tz`` when supplied.

    Carries ``event_id`` and ``field`` (e.g.
    ``events[<event_id>].payload.departure``) so callers can pinpoint the
    offending event.
    """

    def __init__(self, event_id: str, field: str, message: str):
        super().__init__(message)
        self.event_id = event_id
        self.field = field


class UnexpectedEventUserError(TimelineError):
    """A queried timeline event belongs to a different user.

    Raised by ``derive_schedule_state_inputs`` when an event returned by
    ``list_schedule_events`` has ``user_id`` different from the input
    ``user_id``. This is a hard isolation failure; the offending
    ``event_id`` is carried as an attribute.
    """

    def __init__(self, event_id: str, message: str):
        super().__init__(message)
        self.event_id = event_id