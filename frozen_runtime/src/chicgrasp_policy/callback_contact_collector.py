"""Bounded, import-isolated contact callback transport and pure replay reducer.

The injected IPhysxSimulation subscription is called only by ``start``.  Local
intervals bracket the caller's existing simulate/fetch path; they are NOT native
contact timestamps or an invented native callback-completion barrier.
"""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import time

TRANSPORT = "callback-contact/v1"
BOUNDARY_SCHEMA = "gate5-callback-boundary/v1"
LIFECYCLE_SCHEMA = "gate5-callback-lifecycle/v1"
MAX_HEADERS = 4096
MAX_CONTACTS = 16384
MAX_CALLBACKS = 256
MAX_PAIRS = 4096
MAX_LIFECYCLE = 128
HEADER_FIELDS = ("stage_id", "actor0", "actor1", "collider0", "collider1", "type",
                 "num_contact_data", "contact_data_offset", "num_friction_anchors_data",
                 "friction_anchors_offset", "proto_index0", "proto_index1")
_COUNT_KEYS = ("received_callbacks", "received_headers", "received_data", "copied_headers",
               "copied_data", "retained_callbacks", "retained_headers", "retained_data",
               "omitted_callbacks", "omitted_headers", "omitted_data", "copy_failures",
               "attribution_failures", "late_completions", "unknown_header_lengths", "unknown_data_lengths")


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clock(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or not _integer(value[0])
            or value[0] < 0 or not _finite(value[1]) or value[1] < 0):
        raise ValueError("invalid_callback_interval_clock")
    return list(value)


def _error(exc):
    return (type(exc).__name__ + ":" + str(exc))[:512]


def _scalar(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value if not isinstance(value, str) else value[:512]
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    return {"unavailable_native_type": type(value).__name__[:128]}


def _contact(value, index):
    item = {"source_data_index": index, "issues": []}
    for key in ("position", "normal", "impulse"):
        try:
            native = getattr(value, key)
            # Exactly three indexed scalars; never retain a borrowed vector.
            if len(native) != 3:
                raise ValueError("expected_three_scalars")
            vals = [native[i] for i in range(3)]
            item[key + "_raw"] = [_scalar(v) for v in vals]
            if not all(_finite(v) for v in vals):
                item["issues"].append("malformed_or_nonfinite:" + key)
        except Exception as exc:
            item["issues"].append("unavailable:" + key + ":" + _error(exc))
    for key in ("separation", "face_index0", "face_index1", "material0", "material1"):
        try:
            native = getattr(value, key)
            item[key + "_raw"] = _scalar(native)
            if not (_finite(native) if key == "separation" else _integer(native)):
                item["issues"].append("malformed_or_nonfinite:" + key)
        except Exception as exc:
            item["issues"].append("unavailable:" + key + ":" + _error(exc))
    return item


def _header(value, index):
    item = {"source_header_index": index}
    for key in HEADER_FIELDS:
        try:
            native = getattr(value, key)
            if key == "type" and not isinstance(native, (int, bool, float, str)):
                native = int(native)  # ContactEventType is an integer enum.
            item[key] = _scalar(native)
            if not _integer(native):
                item.setdefault("error", "noninteger_native_header:" + key)
        except Exception as exc:
            item.setdefault("error", _error(exc))
    return item


def _counts():
    return {key: 0 for key in _COUNT_KEYS}


def _bucket():
    return {"records": [], "counts": _counts(), "issues": [], "allocated_headers": 0,
            "allocated_data": 0, "allocated_callbacks": 0}


class CallbackContactCollector:
    """Own one supported subscription; copy only local scalar data in callbacks.

    Locks guard bookkeeping, never native subscription/unsubscription or borrowed
    scalar access.  Draining never waits for a copying callback to finish.
    """
    def __init__(self, physics, *, monotonic_ns=time.monotonic_ns,
                 max_headers=MAX_HEADERS, max_contacts=MAX_CONTACTS,
                 max_callbacks=MAX_CALLBACKS, max_lifecycle=MAX_LIFECYCLE):
        limits = (max_headers, max_contacts, max_callbacks, max_lifecycle)
        if any(not _integer(v) or v <= 0 for v in limits):
            raise ValueError("positive_callback_limits_required")
        self.physics, self._now = physics, monotonic_ns
        self.limits = dict(headers=max_headers, data=max_contacts, callbacks=max_callbacks,
                           lifecycle=max_lifecycle, pair_state=MAX_PAIRS)
        self._lock = threading.Lock()
        self._subscription = None
        self._started = self._closed = self._candidate = False
        self._generation = self._epoch = self._sequence = self._interval_sequence = 0
        self._stage_id = None
        self._association = None
        self._initial, self._outside = _bucket(), _bucket()
        self._window = None
        self._inflight = {}
        self._inflight_headers = self._inflight_data = 0
        self._events, self._history = [], []
        self._lifecycle_omitted = 0
        self._totals = _counts()

    def _history_issue(self, issue):
        if issue not in self._history:
            if len(self._history) < 128:
                self._history.append(issue[:512])
            elif "history_issue_metadata_overflow" not in self._history:
                self._history[-1] = "history_issue_metadata_overflow"

    def _event(self, kind, **values):
        event = dict(kind=kind, monotonic_ns=self._now(), generation=self._generation,
                     epoch=self._epoch, **values)
        if len(self._events) < self.limits["lifecycle"]:
            self._events.append(event)
        else:
            self._lifecycle_omitted += 1
            self._history_issue("lifecycle_metadata_overflow")

    def start(self):
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("callback_subscription_already_started_or_closed")
            self._started = True
            self._generation += 1
            generation = self._generation
            self._event("SUBSCRIBE_REQUESTED", setup_point="before ordinary task construction/reset")
        callback = lambda headers, data: self._receive(generation, headers, data)
        try:
            holder = self.physics.subscribe_contact_report_events(callback)
            if holder is None or not callable(getattr(holder, "unsubscribe", None)):
                raise ValueError("supported_subscription_holder_unavailable")
        except Exception as exc:
            with self._lock:
                self._history_issue("subscription_setup_failed")
                self._event("SUBSCRIBE_FAILED", error=_error(exc))
            raise
        with self._lock:
            self._subscription = holder
            self._event("SUBSCRIBE_RETURNED")
        return self

    def associate(self, stage_id, identity_record=None):
        if not _integer(stage_id) or stage_id <= 0:
            raise ValueError("positive_managed_stage_id_required")
        with self._lock:
            if self._stage_id is not None and self._stage_id != stage_id:
                self._new_epoch("managed_stage_replacement")
            self._stage_id = stage_id
            # Caller supplies copied JSON provenance, never native objects.
            self._association = deepcopy(identity_record)
            self._event("MANAGED_ASSOCIATION", stage_id=stage_id,
                        initialization_receipts_relabelled=False)

    def _new_epoch(self, reason):
        if self._window is not None:
            self._window["issues"].append("epoch_changed_during_interval")
        self._epoch += 1
        self._event("EPOCH_CHANGED", reason=str(reason)[:512])

    def new_epoch(self, reason):
        with self._lock:
            self._new_epoch(reason)
        return self._epoch

    def begin_interval(self, clock_before, label=None, *, phase="candidate"):
        before = _clock(clock_before)
        if phase not in ("candidate", "initialization", "unassigned"):
            raise ValueError("invalid_callback_interval_phase")
        with self._lock:
            if not self._started or self._closed or self._subscription is None:
                raise RuntimeError("callback_subscription_not_active")
            if self._window is not None:
                raise RuntimeError("previous_callback_interval_not_drained")
            self._candidate = self._candidate or phase == "candidate"
            self._interval_sequence += 1
            win = _bucket()
            win["interval"] = dict(id=self._interval_sequence, epoch=self._epoch,
                generation=self._generation, phase=phase, label=deepcopy(label),
                clock_before=before, clock_after=None, start_ns=self._now(), completed_ns=None,
                success=None, in_flight_at_completion=[],
                basis="local wrapper of existing step containing simulate then blocking fetch; not native event timestamp")
            self._window = win
            return self._interval_sequence

    def complete_interval(self, clock_after, success=True):
        after = _clock(clock_after)
        with self._lock:
            if self._window is None or self._window["interval"]["completed_ns"] is not None:
                raise RuntimeError("callback_interval_not_open")
            interval = self._window["interval"]
            interval.update(clock_after=after, completed_ns=self._now(), success=success is True)
            interval["in_flight_at_completion"] = sorted(seq for seq, item in self._inflight.items()
                if item["interval_id"] == interval["id"])
            if interval["in_flight_at_completion"]:
                self._window["issues"].append("copy_in_progress_at_step_return")
                self._history_issue("incomplete_callback_copy_observed")
            if success is not True:
                self._window["issues"].append("existing_step_failed")

    def _receive(self, generation, headers, data):
        # The only entry reachable from native code: no decode, scene/pose access,
        # disk access, simulation calls, event producer or cleanup.
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            win = self._window
            active = win is not None and win["interval"]["completed_ns"] is None
            phase = "candidate" if self._candidate else "initialization"
            target = win if active else self._outside if self._candidate or self._closed else self._initial
            record = dict(sequence=sequence, receipt_ns=self._now(), copy_completed_ns=None,
                subscription_generation=generation, epoch=self._epoch,
                interval_id=win["interval"]["id"] if active else None,
                phase=win["interval"]["phase"] if active else phase,
                assignment="PENDING_COPY", raw_header_count=None, raw_data_count=None,
                raw_headers=[], data=[], issues=[], omitted_headers=0, omitted_data=0)
            target["counts"]["received_callbacks"] += 1
            self._totals["received_callbacks"] += 1
            can_store = (target["allocated_callbacks"] < self.limits["callbacks"]
                         and len(self._inflight) < self.limits["callbacks"])
            receipt_metadata_omitted = not can_store
            target["allocated_callbacks"] += 1
            if can_store:
                self._inflight[sequence] = record
            else:
                target["counts"]["omitted_callbacks"] += 1
                self._totals["omitted_callbacks"] += 1
                if "callback_metadata_overflow" not in target["issues"]:
                    target["issues"].append("callback_metadata_overflow")
                self._history_issue("callback_metadata_overflow")
        try:
            if any(v is None or isinstance(v, (str, bytes, dict)) or not hasattr(v, "__getitem__")
                   for v in (headers, data)):
                raise ValueError("malformed_borrowed_contact_vectors")
            nh, nd = len(headers), len(data)
            if not _integer(nh) or not _integer(nd) or nh < 0 or nd < 0:
                raise ValueError("invalid_borrowed_vector_lengths")
            record.update(raw_header_count=nh, raw_data_count=nd)
            with self._lock:
                ch = min(nh, max(0, self.limits["headers"] - target["allocated_headers"]),
                         max(0, self.limits["headers"] - self._inflight_headers)) if can_store else 0
                cd = min(nd, max(0, self.limits["data"] - target["allocated_data"]),
                         max(0, self.limits["data"] - self._inflight_data)) if can_store else 0
                target["allocated_headers"] += ch
                target["allocated_data"] += cd
                self._inflight_headers += ch
                self._inflight_data += cd
                loss_bucket = self._outside if active and self._window is not win else target
                record["length_accounting"] = ("LATE_LENGTHS_FOR_PRIOR_BOUNDARY"
                    if loss_bucket is not target else "RECEIPT_BUCKET")
                for counts in (loss_bucket["counts"], self._totals):
                    counts["received_headers"] += nh
                    counts["received_data"] += nd
                    counts["omitted_headers"] += nh - ch
                    counts["omitted_data"] += nd - cd
                record.update(omitted_headers=nh-ch, omitted_data=nd-cd)
            record["raw_headers"] = [_header(headers[i], i) for i in range(ch)]
            record["data"] = [_contact(data[i], i) for i in range(cd)]
            if nh > ch or nd > cd:
                record["issues"].append("local_callback_payload_overflow")
            if any("error" in h for h in record["raw_headers"]) or any(d["issues"] for d in record["data"]):
                record["issues"].append("borrowed_scalar_copy_invalid")
        except Exception as exc:
            ch = locals().get("ch", 0)
            cd = locals().get("cd", 0)
            record["issues"].append("callback_copy_failed:" + _error(exc))
        with self._lock:
            record["copy_completed_ns"] = self._now()
            copy_failed = any(issue == "borrowed_scalar_copy_invalid" or
                              issue.startswith("callback_copy_failed:") for issue in record["issues"])
            self._inflight_headers -= ch
            self._inflight_data -= cd
            self._inflight.pop(sequence, None)
            missing_headers = ch - len(record["raw_headers"])
            missing_data = cd - len(record["data"])
            if missing_headers or missing_data:
                record["omitted_headers"] += missing_headers
                record["omitted_data"] += missing_data
                loss_bucket = self._outside if active and self._window is not win else target
                for counts in (loss_bucket["counts"], self._totals):
                    counts["omitted_headers"] += missing_headers
                    counts["omitted_data"] += missing_data
            valid = (generation == self._generation and not self._closed and record["epoch"] == self._epoch)
            same_open = (active and self._window is win and win["interval"]["completed_ns"] is None and valid)
            record["assignment"] = ("CURRENT_INTERVAL" if same_open else "INITIALIZATION_UNBOUND"
                if not active and not self._candidate and valid else "LATE_OR_UNASSIGNED")
            if active and not same_open:
                record["issues"].append("copy_completed_outside_original_interval")
            if not valid:
                record["issues"].append("stale_generation_epoch_or_closed")
            if record["assignment"] == "LATE_OR_UNASSIGNED":
                self._history_issue("late_or_unassigned_callback")
            for issue in record["issues"]:
                self._history_issue(issue.split(":", 1)[0])
            # A drained window cannot receive a later mutation. Keep late evidence
            # in the separate bounded outside queue instead, never the next step.
            destination = target
            if active and self._window is not win:
                destination = self._outside
                record["completion_accounting"] = "LATE_COMPLETION_FOR_PRIOR_BOUNDARY"
                destination["counts"]["late_completions"] += 1
                self._totals["late_completions"] += 1
                if len(destination["records"]) >= self.limits["callbacks"]:
                    can_store = False
                    self._history_issue("late_callback_retention_overflow")
                elif (sum(len(r["raw_headers"]) for r in destination["records"]) + len(record["raw_headers"]) > self.limits["headers"]
                      or sum(len(r["data"]) for r in destination["records"]) + len(record["data"]) > self.limits["data"]):
                    can_store = False
                    self._history_issue("late_callback_retention_overflow")
                if not can_store:
                    for counts in (destination["counts"], self._totals):
                        # Receipt metadata may already have been omitted while
                        # this vector's len() was still in progress. Count that
                        # callback once, not again as a late retention loss.
                        counts["omitted_callbacks"] += not receipt_metadata_omitted
                        counts["omitted_headers"] += len(record["raw_headers"])
                        counts["omitted_data"] += len(record["data"])
            for counts in (destination["counts"], self._totals):
                counts["copied_headers"] += len(record["raw_headers"])
                counts["copied_data"] += len(record["data"])
                counts["copy_failures"] += copy_failed
                counts["attribution_failures"] += record["assignment"] == "LATE_OR_UNASSIGNED"
                counts["unknown_header_lengths"] += record["raw_header_count"] is None
                counts["unknown_data_lengths"] += record["raw_data_count"] is None
            if can_store:
                destination["records"].append(record)
                if destination is not target:
                    destination["allocated_callbacks"] += 1
                    destination["allocated_headers"] += len(record["raw_headers"])
                    destination["allocated_data"] += len(record["data"])
                for counts in (destination["counts"], self._totals):
                    counts["retained_callbacks"] += 1
                    counts["retained_headers"] += len(record["raw_headers"])
                    counts["retained_data"] += len(record["data"])

    def take_boundary(self, clock, stage_id):
        now_clock = _clock(clock)
        with self._lock:
            if stage_id != self._stage_id:
                self._new_epoch("observed_boundary_stage_association_mismatch")
            win, outside = self._window, self._outside
            self._window, self._outside = None, _bucket()
            issues = [] if win else ["missing_existing_step_interval"]
            if win:
                issues.extend(win["issues"])
            if outside["records"] or outside["counts"]["received_callbacks"]:
                issues.append("late_or_unassigned_callback")
            issues.extend(outside["issues"])
            if stage_id != self._stage_id:
                issues.append("managed_stage_association_mismatch")
            if self._closed or self._subscription is None:
                issues.append("callback_subscription_not_active")
            envelope = dict(schema=BOUNDARY_SCHEMA, transport=TRANSPORT,
                subscription_generation=self._generation, epoch=self._epoch,
                interval=deepcopy(win["interval"]) if win else None,
                boundary=dict(clock=now_clock, stage_id=stage_id, monotonic_ns=self._now()),
                callbacks=deepcopy(sorted(win["records"], key=lambda r: r["sequence"])) if win else [],
                unassigned_callbacks=deepcopy(sorted(outside["records"], key=lambda r: r["sequence"])),
                counts=deepcopy(win["counts"]) if win else _counts(),
                unassigned_counts=deepcopy(outside["counts"]),
                in_flight_sequences=sorted(self._inflight),
                local_issues=list(dict.fromkeys(issues)), history_issues=list(self._history),
                limits=dict(self.limits), native_drop_accounting="UNAVAILABLE",
                negative_qualification="NOT_ESTABLISHED")
        checked = verify_boundary(envelope)
        envelope["attribution_valid"] = not checked["integrity_issues"] and not checked["admission_issues"]
        return envelope

    def export_lifecycle(self):
        with self._lock:
            result = dict(schema=LIFECYCLE_SCHEMA, transport=TRANSPORT,
                subscription_generation=self._generation, epoch=self._epoch, started=self._started,
                closed=self._closed, subscription_retained=self._subscription is not None,
                stage_id=self._stage_id, association=deepcopy(self._association),
                events=deepcopy(self._events), omitted_lifecycle_records=self._lifecycle_omitted,
                initialization_callbacks=deepcopy(sorted(self._initial["records"], key=lambda r: r["sequence"])),
                initialization_counts=deepcopy(self._initial["counts"]),
                initialization_stage_association=[dict(sequence=r["sequence"], stage_matches=(None
                    if not r["raw_headers"] else self._stage_id is not None and
                    all(h.get("stage_id") == self._stage_id for h in r["raw_headers"])),
                    candidate_current=False) for r in self._initial["records"]],
                undrained_unassigned_callbacks=deepcopy(sorted(self._outside["records"], key=lambda r: r["sequence"])),
                undrained_unassigned_counts=deepcopy(self._outside["counts"]),
                in_flight_sequences=sorted(self._inflight), totals=dict(self._totals),
                history_issues=list(self._history), limits=dict(self.limits),
                negative_qualification="NOT_ESTABLISHED", native_drop_accounting="UNAVAILABLE")
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            holder = self._subscription
            self._event("UNSUBSCRIBE_REQUESTED", holder_present=holder is not None)
        try:
            if holder is not None:
                holder.unsubscribe()
        except Exception as exc:
            with self._lock:
                self._history_issue("unsubscribe_failed")
                self._event("UNSUBSCRIBE_FAILED", error=_error(exc))
            raise
        with self._lock:
            self._subscription = None
            self._event("UNSUBSCRIBE_RETURNED")


def verify_bucket(records, counts, *, in_flight_sequences=(), limits=None):
    """Check original initialization/outside copies without inventing a step.

    Receipt counters stay in the original bucket, completion deltas may be in a
    later outside bucket. Omitted metadata has no invented original record. The
    return distinguishes contradictory persistence from that accounting limit.
    """
    bad, limitations = [], []
    try:
        if not isinstance(records, list) or not isinstance(counts, dict):
            raise ValueError("invalid_callback_bucket")
        if any(not _integer(counts.get(k)) or counts[k] < 0 for k in _COUNT_KEYS):
            raise ValueError("invalid_callback_bucket_counts")
        if not isinstance(in_flight_sequences, (list, tuple)) or any(not _integer(s) for s in in_flight_sequences):
            raise ValueError("invalid_bucket_inflight_sequences")
        bounds = limits or {"callbacks": MAX_CALLBACKS, "headers": MAX_HEADERS, "data": MAX_CONTACTS}
        seqs, raw_h, raw_d, unknown_h, unknown_d = [], 0, 0, 0, 0
        copied_h = copied_d = late = failures = attribution = 0
        for r in records:
            if not isinstance(r, dict) or not _integer(r.get("sequence")):
                raise ValueError("malformed_original_callback_record")
            seqs.append(r["sequence"])
            if (not isinstance(r.get("raw_headers"), list) or not isinstance(r.get("data"), list)
                    or not isinstance(r.get("issues"), list) or any(not isinstance(i, str) for i in r["issues"])):
                raise ValueError("malformed_original_callback_payload")
            if (not _integer(r.get("receipt_ns")) or not _integer(r.get("copy_completed_ns"))
                    or r["copy_completed_ns"] < r["receipt_ns"]):
                bad.append("invalid_original_callback_copy_order")
            is_late = r.get("completion_accounting") == "LATE_COMPLETION_FOR_PRIOR_BOUNDARY"
            late += is_late
            if is_late and r.get("assignment") != "LATE_OR_UNASSIGNED":
                bad.append("late_completion_with_current_assignment")
            if r.get("assignment") not in ("CURRENT_INTERVAL", "INITIALIZATION_UNBOUND", "LATE_OR_UNASSIGNED"):
                bad.append("invalid_original_callback_assignment")
            for key in ("subscription_generation", "epoch"):
                if not _integer(r.get(key)) or r[key] < 0:
                    bad.append("invalid_original_callback_generation")
            failures += any(i == "borrowed_scalar_copy_invalid" or i.startswith("callback_copy_failed:") for i in r["issues"])
            attribution += r.get("assignment") == "LATE_OR_UNASSIGNED"
            copied_h += len(r["raw_headers"]); copied_d += len(r["data"])
            for key, copied, omission in (("raw_header_count", len(r["raw_headers"]), "omitted_headers"),
                                           ("raw_data_count", len(r["data"]), "omitted_data")):
                amount = r.get(key)
                if not _integer(r.get(omission)) or r[omission] < 0:
                    raise ValueError("invalid_original_callback_omission")
                if amount is None:
                    if not any(i.startswith("callback_copy_failed:") for i in r["issues"]):
                        bad.append("unknown_original_length_without_copy_failure")
                    unknown_h += key == "raw_header_count"
                    unknown_d += key == "raw_data_count"
                elif not _integer(amount) or amount < 0:
                    bad.append("invalid_original_callback_length")
                else:
                    if amount != copied + r[omission]:
                        bad.append("original_callback_payload_length_mismatch")
                    if not is_late or r.get("length_accounting") == "LATE_LENGTHS_FOR_PRIOR_BOUNDARY":
                        if key == "raw_header_count":
                            raw_h += amount
                        else:
                            raw_d += amount
            for i, h in enumerate(r["raw_headers"]):
                if not isinstance(h, dict) or h.get("source_header_index") != i:
                    bad.append("original_header_index_mismatch")
                elif any(not _integer(h.get(k)) for k in HEADER_FIELDS) and "error" not in h:
                    bad.append("invalid_original_header_without_copy_issue")
            for i, d in enumerate(r["data"]):
                if not isinstance(d, dict) or d.get("source_data_index") != i or not isinstance(d.get("issues"), list):
                    bad.append("original_data_index_or_issues_mismatch")
                    continue
                invalid = any(not isinstance(d.get(k + "_raw"), list) or len(d[k + "_raw"]) != 3
                              or not all(_finite(v) for v in d[k + "_raw"]) for k in ("position", "normal", "impulse"))
                invalid = invalid or any(not (_finite(d.get(k + "_raw")) if k == "separation" else _integer(d.get(k + "_raw")))
                    for k in ("separation", "face_index0", "face_index1", "material0", "material1"))
                if invalid and (not d["issues"] or "borrowed_scalar_copy_invalid" not in r["issues"]):
                    bad.append("invalid_original_data_without_copy_issue")
        if seqs != sorted(set(seqs)):
            bad.append("original_callback_sequence_duplicate_or_reordered")
        if (counts["retained_callbacks"], counts["retained_headers"], counts["retained_data"]) != (len(records), copied_h, copied_d):
            bad.append("original_retained_counts_mismatch")
        for key, observed in (("received_headers", raw_h), ("received_data", raw_d),
                              ("copied_headers", copied_h), ("copied_data", copied_d),
                              ("late_completions", late), ("copy_failures", failures),
                              ("attribution_failures", attribution), ("unknown_header_lengths", unknown_h),
                              ("unknown_data_lengths", unknown_d)):
            if counts[key] < observed:
                bad.append("original_bucket_counter_below_observed:" + key)
        omitted = any(counts[k] for k in ("omitted_callbacks", "omitted_headers", "omitted_data"))
        if omitted:
            limitations.append("omitted_original_payload_or_metadata_not_reconstructed")
        if in_flight_sequences:
            limitations.append("inflight_copy_not_a_completed_original_record")
        if counts["late_completions"]:
            limitations.append("late_completion_receipts_owned_by_prior_bucket")
        if not counts["omitted_callbacks"] and not in_flight_sequences and late == counts["late_completions"]:
            if counts["received_callbacks"] != len(records) - late:
                bad.append("original_received_callback_count_mismatch")
            for key, observed in (("received_headers", raw_h), ("received_data", raw_d),
                                  ("copied_headers", copied_h), ("copied_data", copied_d),
                                  ("copy_failures", failures), ("attribution_failures", attribution),
                                  ("unknown_header_lengths", unknown_h), ("unknown_data_lengths", unknown_d)):
                if counts[key] != observed:
                    bad.append("original_complete_bucket_counter_mismatch:" + key)
        if len(records) > bounds["callbacks"] or copied_h > bounds["headers"] or copied_d > bounds["data"]:
            bad.append("original_bucket_retention_limit_exceeded")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        bad.append("malformed_callback_bucket:" + str(exc))
    return dict(integrity_issues=list(dict.fromkeys(bad)), accounting_limits=list(dict.fromkeys(limitations)))


def verify_boundary(envelope):
    """Recompute structural integrity and current attribution from saved fields.

    Integrity failure means inconsistent/corrupt persistence. Admission issues can
    be truthful diagnostics (e.g. no callback, overflow, in-flight copy). Neither
    an asserted attribution flag nor an asserted negative flag is trusted.
    """
    bad, unavailable = [], []
    try:
        if envelope.get("schema") != BOUNDARY_SCHEMA or envelope.get("transport") != TRANSPORT:
            raise ValueError("callback_schema_or_transport_mismatch")
        if envelope.get("negative_qualification") != "NOT_ESTABLISHED":
            bad.append("unsupported_negative_qualification")
        if envelope.get("native_drop_accounting") != "UNAVAILABLE":
            bad.append("unsupported_native_drop_accounting")
        boundary, interval = envelope["boundary"], envelope["interval"]
        current = _clock(boundary["clock"])
        if not _integer(boundary["stage_id"]) or boundary["stage_id"] <= 0:
            unavailable.append("invalid_managed_stage_at_boundary")
        if not isinstance(interval, dict):
            unavailable.append("missing_existing_step_interval")
        else:
            before = _clock(interval["clock_before"])
            if interval["clock_after"] is None or interval["completed_ns"] is None:
                unavailable.append("existing_step_interval_not_completed")
            else:
                after = _clock(interval["clock_after"])
                if after != current or after[0] != before[0] + 1 or not after[1] > before[1]:
                    unavailable.append("interval_clock_not_one_completed_step")
            if interval["phase"] != "candidate" or interval["epoch"] != envelope["epoch"] or interval["generation"] != envelope["subscription_generation"]:
                unavailable.append("interval_epoch_generation_or_phase_mismatch")
            if interval["success"] is not True:
                unavailable.append("existing_step_failed")
            times = (interval["start_ns"], interval["completed_ns"], boundary["monotonic_ns"])
            if (not _integer(times[0]) or not _integer(times[2]) or times[0] > times[2]
                    or times[1] is not None and (not _integer(times[1]) or not times[0] <= times[1] <= times[2])):
                bad.append("invalid_interval_monotonic_order")
            if interval["in_flight_at_completion"]:
                unavailable.append("copy_in_progress_at_step_return")
        records = envelope["callbacks"]
        if not isinstance(records, list) or not isinstance(envelope["unassigned_callbacks"], list):
            raise ValueError("invalid_callback_record_container")
        seqs = []
        nh = nd = ch = cd = oh = od = failures = 0
        for r in records:
            seqs.append(r["sequence"])
            if not _integer(r["sequence"]) or not isinstance(r["raw_headers"], list) or not isinstance(r["data"], list):
                raise ValueError("invalid_callback_record")
            if not isinstance(r["issues"], list) or any(not isinstance(i, str) for i in r["issues"]):
                raise ValueError("invalid_callback_issues")
            unavailable.extend(r["issues"])
            if r["raw_header_count"] is None or r["raw_data_count"] is None:
                if not r["issues"]:
                    bad.append("missing_raw_lengths_without_failure")
            else:
                values = (r["raw_header_count"], r["raw_data_count"], r["omitted_headers"], r["omitted_data"])
                if any(not _integer(v) or v < 0 for v in values):
                    raise ValueError("invalid_callback_counts")
                nh += r["raw_header_count"]; nd += r["raw_data_count"]
                oh += r["omitted_headers"]; od += r["omitted_data"]
                if (len(r["raw_headers"]) + r["omitted_headers"] != r["raw_header_count"]
                        or len(r["data"]) + r["omitted_data"] != r["raw_data_count"]):
                    # A throwing vector index can interrupt the copy: explicit
                    # copy failure remains valid evidence, not invented data.
                    if not any(i.startswith("callback_copy_failed:") for i in r["issues"]):
                        bad.append("callback_lengths_do_not_reconcile")
            ch += len(r["raw_headers"]); cd += len(r["data"]); failures += bool(r["issues"])
            occupied = set()
            for hi, h in enumerate(r["raw_headers"]):
                if h.get("source_header_index") != hi:
                    bad.append("callback_header_source_index_mismatch")
                if "error" in h or any(not _integer(h.get(k)) for k in HEADER_FIELDS):
                    unavailable.append("invalid_copied_callback_header")
                    continue
                count, start, kind = h["num_contact_data"], h["contact_data_offset"], h["type"]
                if (kind not in (0, 1, 2) or count < 0 or start < 0
                        or (kind == 1) != (count == 0) or (count and start + count > len(r["data"]))
                        or h["num_friction_anchors_data"] < 0 or h["friction_anchors_offset"] < 0):
                    unavailable.append("invalid_callback_local_contact_range")
                    continue
                slots = set(range(start, start + count))
                if occupied.intersection(slots):
                    unavailable.append("overlapping_callback_contact_ranges")
                occupied.update(slots)
            if len(occupied) != len(r["data"]):
                unavailable.append("unassociated_callback_contact_data")
            for di, item in enumerate(r["data"]):
                if item.get("source_data_index") != di:
                    bad.append("callback_data_source_index_mismatch")
                if not isinstance(item.get("issues"), list):
                    bad.append("missing_callback_contact_copy_issues")
                elif item["issues"]:
                    unavailable.append("invalid_copied_callback_data")
                invalid_payload = False
                for name in ("position", "normal", "impulse"):
                    value = item.get(name + "_raw")
                    if not isinstance(value, list) or len(value) != 3 or not all(_finite(v) for v in value):
                        invalid_payload = True
                for name in ("separation", "face_index0", "face_index1", "material0", "material1"):
                    value = item.get(name + "_raw")
                    if not (_finite(value) if name == "separation" else _integer(value)):
                        invalid_payload = True
                if invalid_payload:
                    unavailable.append("invalid_copied_callback_data")
                    if not item.get("issues") or "borrowed_scalar_copy_invalid" not in r["issues"]:
                        bad.append("invalid_payload_without_recorded_copy_issue")
            if (not interval or r["assignment"] != "CURRENT_INTERVAL"
                    or r["subscription_generation"] != envelope["subscription_generation"]
                    or r["epoch"] != envelope["epoch"] or r["interval_id"] != interval["id"]
                    or r["phase"] != "candidate"):
                unavailable.append("callback_not_current_interval")
            elif (interval["completed_ns"] is None or not _integer(r["receipt_ns"]) or not _integer(r["copy_completed_ns"])
                    or not interval["start_ns"] <= r["receipt_ns"] <= r["copy_completed_ns"] <= interval["completed_ns"]):
                unavailable.append("callback_copy_not_completed_inside_interval")
        if len(seqs) != len(set(seqs)) or seqs != sorted(seqs):
            bad.append("duplicate_or_reordered_callback_sequences")
        counts = envelope["counts"]
        if any(not _integer(counts.get(k)) or counts[k] < 0 for k in _COUNT_KEYS):
            raise ValueError("invalid_boundary_counts")
        if counts["retained_callbacks"] != len(records):
            bad.append("retained_callback_count_mismatch")
        if counts["retained_headers"] != ch or counts["retained_data"] != cd:
            bad.append("retained_payload_count_mismatch")
        if counts["copied_headers"] != ch or counts["copied_data"] != cd:
            if not envelope["in_flight_sequences"]:
                bad.append("copied_payload_count_mismatch")
        if counts["received_callbacks"] != len(records) + counts["omitted_callbacks"]:
            if not envelope["in_flight_sequences"]:
                bad.append("received_callback_count_mismatch")
        if not counts["omitted_callbacks"] and not envelope["in_flight_sequences"]:
            if (counts["received_headers"], counts["received_data"], counts["omitted_headers"], counts["omitted_data"]) != (nh, nd, oh, od):
                bad.append("received_payload_count_mismatch")
        if counts["omitted_callbacks"] or counts["omitted_headers"] or counts["omitted_data"]:
            unavailable.append("local_callback_retention_loss")
        if envelope["in_flight_sequences"]:
            unavailable.append("copy_in_progress_at_boundary")
        if envelope["unassigned_callbacks"] or envelope["unassigned_counts"]["received_callbacks"]:
            unavailable.append("late_or_unassigned_callback")
        local = envelope["local_issues"]
        if not isinstance(local, list) or any(not isinstance(i, str) for i in local):
            raise ValueError("invalid_local_issues")
        unavailable.extend(local)
        limits = envelope["limits"]
        if len(records) > limits["callbacks"] or ch > limits["headers"] or cd > limits["data"]:
            bad.append("saved_callback_limits_exceeded")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        bad.append("malformed_callback_envelope:" + str(exc))
    return dict(integrity_issues=list(dict.fromkeys(bad)), admission_issues=list(dict.fromkeys(unavailable)))


def flatten_boundary(envelope):
    """Flatten copied payloads once, retaining original callback-local offsets."""
    headers, data = [], []
    for r in envelope.get("callbacks", []):
        base = len(data)
        for original in r["data"]:
            item = deepcopy(original)
            item.update(callback_sequence=r["sequence"], callback_source_data_index=original["source_data_index"],
                        source_data_index=len(data))
            data.append(item)
        for original in r["raw_headers"]:
            item = deepcopy(original)
            item.update(callback_sequence=r["sequence"], callback_source_header_index=original["source_header_index"],
                        callback_contact_data_offset=original.get("contact_data_offset"), source_header_index=len(headers))
            if _integer(item.get("contact_data_offset")):
                item["contact_data_offset"] += base
            headers.append(item)
    verification = verify_boundary(envelope)
    return dict(raw_headers=headers, headers=deepcopy(headers), data=data, source_header_count=envelope["counts"]["received_headers"],
                source_data_count=envelope["counts"]["received_data"],
                local_omitted_headers=envelope["counts"]["omitted_headers"],
                local_omitted_data=envelope["counts"]["omitted_data"],
                issues=verification["integrity_issues"] + verification["admission_issues"],
                integrity_issues=verification["integrity_issues"], admission_issues=verification["admission_issues"])


def reduce_boundary(envelope, decoded_headers, coverage, *, awake=None, previous_state=None):
    """Pure replayable event state and same-boundary production witnesses.

    The runtime has no complete production negative certificate. LOST establishes
    ending for that reported pair only; FALSE cannot be forged through a flag.
    """
    verification = verify_boundary(envelope)
    issues = verification["integrity_issues"] + verification["admission_issues"]
    stage_id, epoch = envelope.get("boundary", {}).get("stage_id"), envelope.get("epoch")
    state = {}
    if previous_state and previous_state.get("epoch") == epoch and previous_state.get("stage_id") == stage_id:
        try:
            pairs = previous_state.get("pairs")
            if not isinstance(pairs, list):
                raise ValueError("missing_pair_list")
            if len(pairs) > MAX_PAIRS:
                issues.append("previous_pair_state_capacity_exceeded")
            for pair in pairs[:MAX_PAIRS]:
                key = pair["key"]
                if (not isinstance(key, (tuple, list)) or len(key) != 4 or key[:2] != [stage_id, epoch]
                        or any(not isinstance(slot, (tuple, list)) or len(slot) != 3 for slot in key[2:])):
                    raise ValueError("invalid_pair_key")
                state[tuple(tuple(v) if isinstance(v, list) else v for v in key)] = deepcopy(pair)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append("malformed_previous_pair_state:" + str(exc))
            state = {}
    transitions, grouped, witnesses = [], {}, []
    try:
        flat = flatten_boundary(envelope)
        if not isinstance(decoded_headers, list) or len(decoded_headers) != len(flat["raw_headers"]):
            raise ValueError("decoded_callback_header_coverage_mismatch")
        for index, (raw, h) in enumerate(zip(flat["raw_headers"], decoded_headers)):
            if any(h.get(k) != raw.get(k) for k in HEADER_FIELDS + ("callback_sequence", "source_header_index")):
                raise ValueError("decoded_header_changed_native_fields")
            if "error" in h or h.get("issues"):
                raise ValueError("invalid_decoded_callback_header")
            if any(not _integer(h.get(k)) for k in HEADER_FIELDS):
                raise ValueError("noninteger_callback_header")
            if h["stage_id"] != stage_id or h["type"] not in (0, 1, 2):
                raise ValueError("callback_stage_or_event_type_invalid")
            count, start = h["num_contact_data"], h["contact_data_offset"]
            if count < 0 or start < 0 or (h["type"] == 1) != (count == 0) or (count and start + count > len(flat["data"])):
                raise ValueError("callback_contact_range_invalid")
            if h["proto_index0"] != 0xFFFFFFFF or h["proto_index1"] != 0xFFFFFFFF:
                raise ValueError("instanced_contact_identity_unestablished")
            slots = []
            for side in (0, 1):
                actor, collider = h.get(f"actor{side}_path"), h.get(f"collider{side}_path")
                if not isinstance(actor, str) or not actor.startswith("/") or not isinstance(collider, str) or not collider.startswith("/"):
                    raise ValueError("missing_decoded_contact_identity")
                slots.append((actor, collider, h[f"proto_index{side}"]))
            key = (stage_id, epoch, *sorted(slots))
            jaw_id = None
            for side in (0, 1):
                actor, collider, _ = slots[side]
                other, other_collider, _ = slots[1-side]
                if actor in coverage["jaws"]:
                    jaw = coverage["jaws"][actor]
                    if collider not in jaw["enabled_colliders"]:
                        raise ValueError("unmapped_callback_jaw_collider")
                    if other == coverage["chicken_actor"]:
                        if other_collider not in coverage["chicken_colliders"]:
                            raise ValueError("unmapped_callback_chicken_collider")
                        jaw_id = jaw["jaw_id"]
            grouped.setdefault(key, []).append((h, index, jaw_id))
        for key, events in grouped.items():
            kinds = {h["type"] for h, _, _ in events}
            if 1 in kinds and kinds != {1}:
                issues.append("unsupported_conflicting_pair_transitions")
                state.pop(key, None)
                transitions.append(dict(key=list(key), transition="CONFLICT", header_indices=[i for _, i, _ in events]))
                continue
            kind = "LOST" if kinds == {1} else "FOUND" if 0 in kinds else "PERSIST"
            transitions.append(dict(key=list(key), transition=kind, header_indices=[i for _, i, _ in events]))
            if kind == "LOST":
                state.pop(key, None)
                continue
            if key not in state and len(state) >= MAX_PAIRS:
                issues.append("pair_state_capacity_exceeded")
                continue
            state[key] = dict(key=list(key), last_interval_id=envelope.get("interval", {}).get("id"), event=kind)
            for h, i, jaw_id in events:
                if jaw_id is None:
                    continue
                involved = (h["actor0_path"], h["actor1_path"])
                if not isinstance(awake, dict) or any(awake.get(actor) is not True for actor in involved):
                    issues.append("awake_currentness_unavailable")
                    continue
                witnesses.append(dict(jaw_id=jaw_id, key=list(key), header_index=i,
                    callback_sequence=h["callback_sequence"], event=kind))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        issues.append(str(exc))
    active = sorted({w["jaw_id"] for w in witnesses})
    current = bool(set(active) & {1, 2}) and bool(set(active) & {3, 4})
    issues = list(dict.fromkeys(issues))
    status = "TRUE" if current and not issues else "UNAVAILABLE"
    if not current and not issues:
        issues.append("negative_contact_qualification_not_established")
    return dict(schema="gate5-callback-currentness/v1", transport=TRANSPORT, status=status,
        left=bool(set(active) & {1, 2}) if not issues else None,
        right=bool(set(active) & {3, 4}) if not issues else None,
        active_jaw_ids=active if status == "TRUE" else [], witnesses=witnesses,
        pair_transitions=transitions, issues=issues,
        negative_qualification="NOT_ESTABLISHED",
        ending_validation="REPORTED_PAIR_ENDING_OBSERVED" if any(t["transition"] == "LOST" for t in transitions) else "NOT_ESTABLISHED",
        state=dict(stage_id=stage_id, epoch=epoch, pairs=list(state.values())),
        history_issues=deepcopy(envelope.get("history_issues", [])))
