"""Pilot close interpretation. Historical owner mechanics remain untouched."""
from copy import deepcopy


def classify_cleanup(before, closed):
    """Never confuse a never-requested no-op with missing ownership after request."""
    issues=list(closed.get('issues',[]))
    cleanup=closed.get('cleanup')
    if not isinstance(cleanup,dict):
        return dict(status='ERROR',issues=issues+['cleanup_record_unavailable'])
    issues+=list(cleanup.get('issues',[]))+list(cleanup.get('lifecycle_issues',[]))
    if issues:return dict(status='ERROR',issues=issues)
    if cleanup.get('cleanup_confirmed') is True:
        return dict(status='CLEANUP_CONFIRMED',issues=[],idempotent=bool(cleanup.get('idempotent',False)))
    attachment=before.get('attachment',{})
    operations=before.get('operations')
    never_requested=(before.get('capture_policy')=='next_scheduled_substep'
        and before.get('held') is False and attachment.get('status')=='UNATTACHED'
        and all(attachment.get(k) is None for k in ('joint_path','generation','request_context','confirmation_context'))
        and isinstance(operations,list)
        and all(op.get('kind') in ('release','lifecycle','close') and not op.get('issues') for op in operations)
        and all(op.get('no_owned_resource') is True and op.get('cleanup_required') is False
                for op in operations if op.get('kind')=='release'))
    if (never_requested and cleanup.get('no_owned_resource') is True
            and cleanup.get('cleanup_required') is False
            and cleanup.get('usd_removed') is False
            and cleanup.get('native_destruction_observed') is False):
        return dict(status='NO_ATTACHMENT_NOT_REQUIRED',issues=[],idempotent=True,
                    note='Known never-requested no-op; no removal/destruction measurement claimed.')
    return dict(status='ERROR',issues=['owner_cleanup_unconfirmed_or_ownership_uncertain'])


