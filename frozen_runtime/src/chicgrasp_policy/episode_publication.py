"""Bounded episode directory publication. No simulator imports or cleanup."""
from datetime import datetime, timezone
import ctypes
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid


def destination_absent(path):
    # lstat also detects dangling links; inaccessible is not absent.
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(f'Episode destination conflict: {path}')


def invocation_deadline():
    """Convert the parent's fixed Windows QPC deadline conservatively once."""
    ticks = os.environ.get('CHICGRASP_PUBLICATION_DEADLINE_QPC')
    frequency = os.environ.get('CHICGRASP_PUBLICATION_QPC_FREQUENCY')
    if ticks is None and frequency is None:
        return None  # ordinary offline callers have no enclosing invocation
    if os.name != 'nt' or ticks is None or frequency is None:
        raise ValueError('invalid_publication_invocation_deadline')
    before = time.monotonic()
    counter, actual_frequency = ctypes.c_longlong(), ctypes.c_longlong()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel.QueryPerformanceCounter(ctypes.byref(counter)) or not kernel.QueryPerformanceFrequency(ctypes.byref(actual_frequency)):
        raise ctypes.WinError(ctypes.get_last_error())
    if int(frequency) != actual_frequency.value or int(frequency) <= 0:
        raise ValueError('publication_deadline_clock_mismatch')
    return before + (int(ticks) - counter.value) / actual_frequency.value


def error_record(exc):
    trace = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return dict(type=type(exc).__name__, message=str(exc), winerror=getattr(exc, 'winerror', None),
                errno=getattr(exc, 'errno', None), traceback=trace[:8192], traceback_truncated=len(trace)>8192)


def write_diagnostic(path, record):
    # Append snapshots: an interrupted update cannot erase the previous receipt.
    # Each snapshot is flushed and closed before a rename or wait.
    with path.open('ab') as stream:
        data=(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode()
        if stream.write(data) != len(data):
            raise OSError('incomplete_publication_diagnostic_write')
        stream.flush()
        os.fsync(stream.fileno())


class Publication:
    def __init__(self, destination, *, deadline=None, clock=None, sleep=None):
        self.destination = Path(destination)
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.invocation_deadline = invocation_deadline() if deadline is None else deadline
        if self.invocation_deadline is not None and not math.isfinite(self.invocation_deadline):
            raise ValueError('nonfinite_publication_deadline')
        attempt = uuid.uuid4().hex
        self.path = self.destination.parent / f'.{self.destination.name}.publication-{attempt}.jsonl'
        self.source = None
        self.record = dict(schema='episode-publication/v1', attempt_id=attempt,
            destination=str(self.destination), source=None, published=False, status='PREPARING',
            attempts=0, retry_count=0, first_rename_monotonic=None, retry_deadline_monotonic=None,
            invocation_deadline_monotonic=self.invocation_deadline, writer_close_outcomes=[],
            mappings='NONE_CREATED', directory_iterators='CLOSED_BEFORE_RENAME',
            handle_owners=dict(status='UNAVAILABLE', reason='No configured bounded read-only handle-owner collector'),
            errors=[], diagnostic_errors=[], diagnostics_incomplete=False, diagnostics_truncated=False)

    def snapshot(self):
        entries=[]
        # Bound metadata volume; never recursively traverse links or payloads.
        base = self.destination if self.record['published'] else self.source
        for parent in [base, base/'policy' if base else None]:
            if parent is None:
                continue
            try:
                with os.scandir(parent) as iterator:
                    for index, item in enumerate(iterator):
                        if index >= 32:
                            self.record['diagnostics_truncated']=True
                            break
                        try:
                            stat=item.stat(follow_symlinks=False)
                            entries.append(dict(path=item.path, bytes=stat.st_size, mode=stat.st_mode,
                                mtime_ns=stat.st_mtime_ns, attributes=getattr(stat,'st_file_attributes',None)))
                        except OSError as exc:
                            entries.append(dict(path=item.path,error=error_record(exc)))
                            self.record['diagnostics_incomplete']=True
            except OSError as exc:
                entries.append(dict(path=str(parent),error=error_record(exc)))
                if not self.record['published']:
                    self.record['diagnostics_incomplete']=True
        self.record['directory_listing']=entries
        try:
            stat=self.destination.lstat()
            self.record['destination_metadata']=dict(state='EXISTS',mode=stat.st_mode,bytes=stat.st_size)
        except FileNotFoundError:
            self.record['destination_metadata']=dict(state='ABSENT')
        except OSError as exc:
            self.record['destination_metadata']=dict(state='UNKNOWN',error=error_record(exc))
            self.record['diagnostics_incomplete']=True

    def persist(self):
        try:
            self.snapshot()
            self.record.update(utc=datetime.now(timezone.utc).isoformat(), monotonic=self.clock())
            write_diagnostic(self.path,self.record)
            return True
        except Exception as exc:
            self.record['diagnostics_incomplete']=True
            self.record['diagnostic_errors'].append(error_record(exc))
            return False

    def failed(self, exc):
        self.record['status']='FAILED'
        self.record['errors'].append(error_record(exc))
        self.persist()  # best effort; never substitute for the primary failure
        exc.publication_diagnostics=self.record
        exc.publication_diagnostic_path=str(self.path)

    def write_payload(self, path, data):
        outcome=dict(path=str(path), opened=False, flushed=False, closed=False, errors=[])
        self.record['writer_close_outcomes'].append(outcome)
        stream=None
        primary=None
        try:
            stream=path.open('xb');outcome['opened']=True
            if stream.write(data)!=len(data):
                raise OSError(f'Incomplete write: {path}')
            stream.flush();outcome['flushed']=True
        except BaseException as exc:
            primary=exc;outcome['errors'].append(error_record(exc))
        finally:
            if stream is not None:
                try:
                    stream.close();outcome['closed']=True
                except BaseException as exc:
                    outcome['errors'].append(error_record(exc))
                    if primary is None:
                        primary=exc
        if primary is not None:
            raise primary

    def publish(self, staging):
        self.source=staging;self.record['source']=str(staging)
        self.record['status']='READY_TO_PUBLISH'
        if not self.persist():
            raise OSError('publication_diagnostic_unavailable_before_rename')
        first=None
        last=None
        while True:
            destination_absent(self.destination)
            now=self.clock()
            if first is None:
                if self.invocation_deadline is not None and now>=self.invocation_deadline:
                    raise TimeoutError('publication_invocation_deadline_exhausted')
                first=now
                deadline=min(first+5.0,self.invocation_deadline) if self.invocation_deadline is not None else first+5.0
                self.record.update(first_rename_monotonic=first,retry_deadline_monotonic=deadline)
            elif now>=deadline:
                raise last
            self.record['attempts']+=1
            self.record['retry_count']=self.record['attempts']-1
            try:
                # Windows Path.rename uses non-replacing os.rename. No copy/replace.
                staging.rename(self.destination)
            except OSError as exc:
                last=exc
                self.record['errors'].append(error_record(exc))
                try:
                    destination_absent(self.destination)
                except OSError as conflict:
                    raise conflict from exc
                if getattr(exc,'winerror',None) not in (5,32,33):
                    raise
                self.record['status']='RETRY_PENDING'
                if not self.persist():
                    raise exc  # diagnostics must be closed/persisted before retries
                remaining=deadline-self.clock()
                if remaining<=0:
                    raise
                self.sleep(min(.250,remaining))
                continue
            self.record.update(status='PUBLISHED',published=True)
            if not self.persist():
                raise OSError('publication_diagnostic_failed_after_rename')
            return self.destination.resolve()
