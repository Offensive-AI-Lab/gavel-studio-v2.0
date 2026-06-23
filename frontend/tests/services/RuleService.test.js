// Behavior tests for the RuleService publish flows.
//
// RuleService.js is a pure service module (no React component) exporting
// `publishDraftRule` and `publishDraftCE`. Both orchestrate:
//   - a force library sync
//   - a name-conflict probe (checkLibraryName) that, when the name is taken,
//     opens a 3-way modal (bookmark / rename / cancel)
//   - the actual publish call + an outcome modal
//
// We mock '../api' (every export the module touches), 'sweetalert2' (so we
// can script the modal decisions deterministically), and the ConfirmDialog
// helpers (so alert/confirm dialogs don't hit Swal directly). PipelineModal
// is left real — it's pure presentation rendered via renderToStaticMarkup.

import { describe, it, expect, vi, beforeEach } from 'vitest';

// --- API mock. Cover every export RuleService imports/uses. ---
vi.mock('../../src/api', () => {
    const ok = (data = {}) => Promise.resolve({ data });
    return {
        default: {
            get: vi.fn(() => ok()),
            post: vi.fn(() => ok()),
        },
        publishCE: vi.fn(() => ok({ status: 'success' })),
        publishRule: vi.fn(() => ok({ status: 'success' })),
        checkLibraryName: vi.fn(() => ok({ exists: false })),
        addRuleBookmark: vi.fn(() => ok()),
        addCEBookmark: vi.fn(() => ok()),
    };
});

// --- Swal mock. fire() returns whatever the current script says. ---
vi.mock('sweetalert2', () => ({
    default: {
        fire: vi.fn(() => Promise.resolve({ isConfirmed: true })),
        close: vi.fn(),
        getPopup: vi.fn(() => null),
    },
}));

// --- ConfirmDialog helpers — used for alert dialogs in the flows. ---
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: vi.fn(() => Promise.resolve()),
    showConfirmDialog: vi.fn(() => Promise.resolve(true)),
}));

import Swal from 'sweetalert2';
import api, {
    publishCE,
    publishRule,
    checkLibraryName,
    addRuleBookmark,
    addCEBookmark,
} from '../../src/api';
import { showAlertDialog } from '../../src/components/ConfirmDialog/confirmDialog';
import { publishDraftRule, publishDraftCE } from '../../src/services/RuleService';

// Helper: queue Swal.fire return values one per call, in order.
const scriptSwal = (...results) => {
    Swal.fire.mockReset();
    let i = 0;
    Swal.fire.mockImplementation(() => {
        const r = results[Math.min(i, results.length - 1)];
        i += 1;
        return Promise.resolve(r ?? { isConfirmed: true });
    });
};

const okData = (data = {}) => Promise.resolve({ data });

beforeEach(() => {
    vi.clearAllMocks();
    // Reset default resolved values (clearAllMocks clears implementations set
    // via mockImplementation but keeps the vi.fn; re-arm sensible defaults).
    api.get.mockImplementation(() => okData());
    api.post.mockImplementation(() => okData());
    publishRule.mockImplementation(() => okData({ status: 'success' }));
    publishCE.mockImplementation(() => okData({ status: 'success' }));
    checkLibraryName.mockImplementation(() => okData({ exists: false }));
    addRuleBookmark.mockImplementation(() => Promise.resolve());
    addCEBookmark.mockImplementation(() => Promise.resolve());
    Swal.fire.mockImplementation(() => Promise.resolve({ isConfirmed: true }));
    Swal.close.mockImplementation(() => {});
    Swal.getPopup.mockImplementation(() => null);
});

// ===========================================================================
// publishDraftRule
// ===========================================================================
describe('publishDraftRule', () => {
    it('aborts and alerts when the rule has no underlying id', async () => {
        await publishDraftRule({ custom_name: 'X' }, 1, vi.fn());
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Cannot publish yet', variant: 'info' }),
        );
        expect(api.get).not.toHaveBeenCalled();
        expect(publishRule).not.toHaveBeenCalled();
    });

    it('uses source_rule_id when present', async () => {
        await publishDraftRule({ source_rule_id: 'src-1', rule_id: 'r-1', custom_name: 'N' }, 1, vi.fn());
        expect(publishRule).toHaveBeenCalledWith('src-1');
    });

    it('falls back to rule_id when source_rule_id is absent', async () => {
        await publishDraftRule({ rule_id: 'r-9', custom_name: 'N' }, 1, vi.fn());
        expect(publishRule).toHaveBeenCalledWith('r-9');
    });

    it('happy path with no name conflict: force-syncs, publishes, auto-bookmarks, shows result, refreshes', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: false }));
        publishRule.mockImplementation(() => okData({ status: 'success', name: 'My Rule' }));
        const refresh = vi.fn();

        await publishDraftRule({ rule_id: 'r-1', custom_name: 'My Rule' }, 42, refresh);

        // pre-publish force-sync
        expect(api.get).toHaveBeenCalledWith('/library/sync', { params: { force: true } });
        expect(checkLibraryName).toHaveBeenCalledWith({ kind: 'rule', name: 'My Rule' });
        expect(publishRule).toHaveBeenCalledWith('r-1');
        // auto-bookmark on success with a userId
        expect(addRuleBookmark).toHaveBeenCalledWith(42, 'r-1');
        // outcome modal + refresh
        expect(Swal.close).toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
    });

    it('continues when the pre-publish sync fails', async () => {
        api.get.mockImplementation((url) => {
            if (url === '/library/sync') return Promise.reject(new Error('sync down'));
            return okData();
        });
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 1, vi.fn());
        expect(publishRule).toHaveBeenCalledWith('r-1');
    });

    it('does not auto-bookmark when no userId is supplied', async () => {
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, null, vi.fn());
        expect(addRuleBookmark).not.toHaveBeenCalled();
    });

    it('does not auto-bookmark when publish is not successful', async () => {
        publishRule.mockImplementation(() => okData({ status: 'conflict', conflict_with: { name: 'N', type: 'rule' } }));
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 7, vi.fn());
        expect(addRuleBookmark).not.toHaveBeenCalled();
    });

    it('builds an error result and still shows outcome modal when publish throws', async () => {
        publishRule.mockImplementation(() => Promise.reject({ response: { data: { detail: 'boom' } } }));
        const refresh = vi.fn();
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 1, refresh);
        // The error path closes the toast, shows the result modal, refreshes.
        expect(Swal.close).toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
        expect(addRuleBookmark).not.toHaveBeenCalled();
    });

    it('survives publish throwing a bare error with no response', async () => {
        publishRule.mockImplementation(() => Promise.reject(new Error('network')));
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 1, vi.fn());
        expect(Swal.close).toHaveBeenCalled();
    });

    it('tolerates a missing refreshData callback', async () => {
        await expect(
            publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 1, undefined),
        ).resolves.toBeUndefined();
    });

    it('continues to publish when the name probe itself throws', async () => {
        checkLibraryName.mockImplementation(() => Promise.reject(new Error('probe failed')));
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'N' }, 1, vi.fn());
        expect(publishRule).toHaveBeenCalledWith('r-1');
    });

    // ---- name-conflict branches ----
    it('cancels cleanly when the conflict modal returns cancel', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        // Top modal: not confirmed, not denied => cancel
        scriptSwal({ isConfirmed: false, isDenied: false });
        const refresh = vi.fn();
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, refresh);
        expect(publishRule).not.toHaveBeenCalled();
        expect(refresh).not.toHaveBeenCalled();
    });

    it('bookmark path: discards draft, syncs, resolves local id, bookmarks, refreshes', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: { predicate: 'p' } }));
        // Top modal: confirmed => bookmark existing
        scriptSwal({ isConfirmed: true });
        api.get.mockImplementation((url) => {
            if (url.startsWith('/library/record/rule/')) return okData({ summary: { local_id: 'loc-9' } });
            return okData();
        });
        const refresh = vi.fn();

        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 5, refresh);

        expect(api.post).toHaveBeenCalledWith('/ai/discard-pipeline-resources', { ce_ids: [], rule_id: 'r-1' });
        expect(addRuleBookmark).toHaveBeenCalledWith(5, 'loc-9');
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Saved', variant: 'success' }));
        expect(publishRule).not.toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
    });

    it('bookmark path: errors when the local record cannot be resolved after sync', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal({ isConfirmed: true });
        api.get.mockImplementation((url) => {
            if (url.startsWith('/library/record/rule/')) return okData({ summary: {} }); // no local_id
            return okData();
        });
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 5, vi.fn());
        expect(addRuleBookmark).not.toHaveBeenCalled();
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Could not bookmark', variant: 'error' }));
    });

    it('bookmark path: tolerates discard + sync failures and still attempts bookmark', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal({ isConfirmed: true });
        api.post.mockImplementation(() => Promise.reject(new Error('discard down')));
        api.get.mockImplementation((url) => {
            if (url === '/library/sync') return Promise.reject(new Error('sync down'));
            if (url.startsWith('/library/record/rule/')) return okData({ summary: { local_id: 'loc-1' } });
            return okData();
        });
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 5, vi.fn());
        expect(addRuleBookmark).toHaveBeenCalledWith(5, 'loc-1');
    });

    it('rename path: persists the new name then publishes', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} })) // initial probe
            .mockImplementation(() => okData({ exists: false })); // rename live-check
        // Top modal: denied => rename; then rename-input modal confirms with a new value.
        scriptSwal(
            { isConfirmed: false, isDenied: true },
            { isConfirmed: true, value: 'New Name' },
        );
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        expect(api.post).toHaveBeenCalledWith('/ai/rename-rule', { rule_id: 'r-1', new_name: 'New Name' });
        expect(publishRule).toHaveBeenCalledWith('r-1');
    });

    it('rename path: loops when the chosen name is also taken, then accepts a free one', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} })) // initial probe
            .mockImplementationOnce(() => okData({ exists: true })) // first candidate taken
            .mockImplementation(() => okData({ exists: false })); // second candidate free
        scriptSwal(
            { isConfirmed: false, isDenied: true },            // top: rename
            { isConfirmed: true, value: 'Taken2' },            // first candidate (taken)
            { isConfirmed: true, value: 'FreeName' },          // second candidate (free)
        );
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        // "Also taken" alert shown once
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Also taken' }));
        expect(api.post).toHaveBeenCalledWith('/ai/rename-rule', { rule_id: 'r-1', new_name: 'FreeName' });
    });

    it('rename path: cancelling the rename input aborts the whole flow', async () => {
        checkLibraryName.mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal(
            { isConfirmed: false, isDenied: true }, // top: rename
            { isConfirmed: false },                 // rename input cancelled
        );
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        expect(api.post).not.toHaveBeenCalledWith('/ai/rename-rule', expect.anything());
        expect(publishRule).not.toHaveBeenCalled();
    });

    it('rename path: continues past a failed live name-check probe', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} })) // initial probe
            .mockImplementationOnce(() => Promise.reject(new Error('probe down'))); // live-check throws -> proceed
        scriptSwal(
            { isConfirmed: false, isDenied: true },
            { isConfirmed: true, value: 'New Name' },
        );
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        expect(api.post).toHaveBeenCalledWith('/ai/rename-rule', { rule_id: 'r-1', new_name: 'New Name' });
    });

    it('rename path: aborts when persisting the new name fails', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} }))
            .mockImplementation(() => okData({ exists: false }));
        scriptSwal(
            { isConfirmed: false, isDenied: true },
            { isConfirmed: true, value: 'New Name' },
        );
        api.post.mockImplementation((url) => {
            if (url === '/ai/rename-rule') return Promise.reject({ response: { data: { detail: 'no' } } });
            return okData();
        });
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Rename failed', variant: 'error' }));
        expect(publishRule).not.toHaveBeenCalled();
    });

    it('renders a rich conflict preview when summary has predicate/description/categories', async () => {
        checkLibraryName.mockImplementation(() => okData({
            exists: true,
            public_id: 'pub-1',
            summary: { predicate: 'x > 0', description: 'desc', categories: ['A', 'B'] },
        }));
        scriptSwal({ isConfirmed: false, isDenied: false }); // cancel
        await publishDraftRule({ rule_id: 'r-1', custom_name: 'Taken' }, 1, vi.fn());
        // The top conflict modal fired with html built from the summary preview.
        const html = Swal.fire.mock.calls[0][0].html;
        expect(html).toEqual(expect.stringContaining('x &gt; 0'));
    });
});

// ===========================================================================
// publishDraftCE
// ===========================================================================
describe('publishDraftCE', () => {
    it('aborts and alerts when the CE has no id', async () => {
        await publishDraftCE({ name: 'X' }, 1, vi.fn());
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Cannot publish yet', variant: 'info' }),
        );
        expect(publishCE).not.toHaveBeenCalled();
    });

    it('happy path with no conflict: force-syncs, publishes, auto-bookmarks, refreshes', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: false }));
        publishCE.mockImplementation(() => okData({ status: 'success', name: 'My CE' }));
        const refresh = vi.fn();

        await publishDraftCE({ ce_id: 'ce-1', name: 'My CE' }, 3, refresh);

        expect(api.get).toHaveBeenCalledWith('/library/sync', { params: { force: true } });
        expect(checkLibraryName).toHaveBeenCalledWith({ kind: 'ce', name: 'My CE' });
        expect(publishCE).toHaveBeenCalledWith('ce-1');
        expect(addCEBookmark).toHaveBeenCalledWith(3, 'ce-1');
        expect(Swal.close).toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
    });

    it('continues when the pre-publish sync fails', async () => {
        api.get.mockImplementation((url) => {
            if (url === '/library/sync') return Promise.reject(new Error('down'));
            return okData();
        });
        await publishDraftCE({ ce_id: 'ce-1', name: 'N' }, 1, vi.fn());
        expect(publishCE).toHaveBeenCalledWith('ce-1');
    });

    it('does not auto-bookmark without a userId', async () => {
        await publishDraftCE({ ce_id: 'ce-1', name: 'N' }, null, vi.fn());
        expect(addCEBookmark).not.toHaveBeenCalled();
    });

    it('does not auto-bookmark when publish is unsuccessful', async () => {
        publishCE.mockImplementation(() => okData({ status: 'race' }));
        await publishDraftCE({ ce_id: 'ce-1', name: 'N' }, 2, vi.fn());
        expect(addCEBookmark).not.toHaveBeenCalled();
    });

    it('builds an error result when publish throws', async () => {
        publishCE.mockImplementation(() => Promise.reject({ response: { data: { detail: 'kaput' } } }));
        await publishDraftCE({ ce_id: 'ce-1', name: 'N' }, 1, vi.fn());
        expect(Swal.close).toHaveBeenCalled();
    });

    it('continues to publish when the name probe throws', async () => {
        checkLibraryName.mockImplementation(() => Promise.reject(new Error('probe down')));
        await publishDraftCE({ ce_id: 'ce-1', name: 'N' }, 1, vi.fn());
        expect(publishCE).toHaveBeenCalledWith('ce-1');
    });

    it('conflict cancel: nothing is published', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal({ isConfirmed: false, isDenied: false });
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 1, vi.fn());
        expect(publishCE).not.toHaveBeenCalled();
    });

    it('bookmark path: discards (with ce id), syncs, resolves local id, bookmarks', async () => {
        checkLibraryName.mockImplementation(() => okData({
            exists: true,
            public_id: 'pub-1',
            summary: { definition: 'd', category: 'cat' },
        }));
        scriptSwal({ isConfirmed: true });
        api.get.mockImplementation((url) => {
            if (url.startsWith('/library/record/ce/')) return okData({ summary: { local_id: 'loc-2' } });
            return okData();
        });
        const refresh = vi.fn();

        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 8, refresh);

        expect(api.post).toHaveBeenCalledWith('/ai/discard-pipeline-resources', { ce_ids: ['ce-1'], rule_id: null });
        expect(addCEBookmark).toHaveBeenCalledWith(8, 'loc-2');
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Saved', variant: 'success' }));
        expect(publishCE).not.toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
    });

    it('bookmark path: errors when local CE cannot be resolved', async () => {
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal({ isConfirmed: true });
        api.get.mockImplementation((url) => {
            if (url.startsWith('/library/record/ce/')) return okData({ summary: {} });
            return okData();
        });
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 8, vi.fn());
        expect(addCEBookmark).not.toHaveBeenCalled();
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Could not bookmark', variant: 'error' }));
    });

    it('rename path: persists the new CE name then publishes', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} }))
            .mockImplementation(() => okData({ exists: false }));
        scriptSwal(
            { isConfirmed: false, isDenied: true },
            { isConfirmed: true, value: 'New CE' },
        );
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 1, vi.fn());
        expect(api.post).toHaveBeenCalledWith('/ai/rename-ce', { ce_id: 'ce-1', new_name: 'New CE' });
        expect(publishCE).toHaveBeenCalledWith('ce-1');
    });

    it('rename path: aborts when the CE rename request fails', async () => {
        checkLibraryName
            .mockImplementationOnce(() => okData({ exists: true, public_id: 'pub-1', summary: {} }))
            .mockImplementation(() => okData({ exists: false }));
        scriptSwal(
            { isConfirmed: false, isDenied: true },
            { isConfirmed: true, value: 'New CE' },
        );
        api.post.mockImplementation((url) => {
            if (url === '/ai/rename-ce') return Promise.reject(new Error('nope'));
            return okData();
        });
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 1, vi.fn());
        expect(showAlertDialog).toHaveBeenCalledWith(expect.objectContaining({ title: 'Rename failed', variant: 'error' }));
        expect(publishCE).not.toHaveBeenCalled();
    });

    it('bookmark path: tolerates discard + sync failures and still bookmarks the CE', async () => {
        // Covers the CE-side discard-failure and sync-failure catch branches.
        checkLibraryName.mockImplementation(() => okData({ exists: true, public_id: 'pub-1', summary: {} }));
        scriptSwal({ isConfirmed: true }); // bookmark existing
        api.post.mockImplementation(() => Promise.reject(new Error('discard down')));
        api.get.mockImplementation((url) => {
            if (url === '/library/sync') return Promise.reject(new Error('sync down'));
            if (url.startsWith('/library/record/ce/')) return okData({ summary: { local_id: 'loc-7' } });
            return okData();
        });
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 4, vi.fn());
        expect(api.post).toHaveBeenCalledWith('/ai/discard-pipeline-resources', { ce_ids: ['ce-1'], rule_id: null });
        expect(addCEBookmark).toHaveBeenCalledWith(4, 'loc-7');
    });

    it('swallows a CE auto-bookmark failure after a successful publish', async () => {
        // Covers the best-effort auto-bookmark catch on the CE happy path.
        publishCE.mockImplementation(() => okData({ status: 'success', name: 'My CE' }));
        addCEBookmark.mockImplementation(() => Promise.reject(new Error('bookmark down')));
        const refresh = vi.fn();
        await publishDraftCE({ ce_id: 'ce-1', name: 'My CE' }, 9, refresh);
        // Despite the bookmark failure, the flow completes: outcome modal + refresh.
        expect(addCEBookmark).toHaveBeenCalledWith(9, 'ce-1');
        expect(Swal.close).toHaveBeenCalled();
        expect(refresh).toHaveBeenCalled();
    });

    it('renders a rich CE conflict preview from definition/category/categories', async () => {
        // Exercises the CE branch of the conflict-preview builder.
        checkLibraryName.mockImplementation(() => okData({
            exists: true,
            public_id: 'pub-1',
            summary: { definition: 'a CE def', category: 'Safety', categories: ['X', 'Y'] },
        }));
        scriptSwal({ isConfirmed: false, isDenied: false }); // cancel
        await publishDraftCE({ ce_id: 'ce-1', name: 'Taken' }, 1, vi.fn());
        const html = Swal.fire.mock.calls[0][0].html;
        expect(html).toEqual(expect.stringContaining('a CE def'));
        expect(html).toEqual(expect.stringContaining('Safety'));
    });
});
