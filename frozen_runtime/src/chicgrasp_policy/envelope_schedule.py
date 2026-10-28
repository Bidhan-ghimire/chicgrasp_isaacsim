"""Integer-only learned envelope selection. No runtime or admission predicates."""


def event_schedule(c, *, protocol="v2"):
    if protocol not in ("v2", "v3c"):
        raise ValueError("unknown_learned_envelope_protocol")
    if type(c) is not int or not 0 <= c <= 199:
        raise ValueError("eligible_accepted_close_required")
    a = 12*c
    start, decision, confirmation = (109, 169, 170) if protocol == "v3c" else (179, 239, 240)
    return dict(close_action=c, anchor=a, admission_start=a+start, decision=a+decision,
        confirmation=a+confirmation, rise_end=a+719, room_reference=a+719,
        room_start=a+720, score_end=a+959, total_actions=max(200, c+80),
        admission_count=61, rise_count=720-confirmation, room_count=240,
        tail_count=max(0, 12*(120-c)))
