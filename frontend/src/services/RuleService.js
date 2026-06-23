import React from 'react';
import Swal from 'sweetalert2';
import { renderToStaticMarkup } from 'react-dom/server';
import api, {
    publishCE,
    publishRule,
    publishRuleSet,
    adoptPublicCE,
    checkLibraryName,
    addRuleBookmark,
    addCEBookmark,
} from '../api';
import PipelineModal from '../components/PipelineModal/PipelineModal';
import { showAlertDialog, showConfirmDialog } from '../components/ConfirmDialog/confirmDialog';

const escapeHtml = (value) => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const escapeJoined = (items, separator = ', ') => (items || []).map((item) => escapeHtml(item)).join(separator);

const h = React.createElement;
const renderPipelineHtml = (props) => renderToStaticMarkup(h(PipelineModal, props));

const showProgressModal = (title, subtitle, options = {}) => {
    const { showSpinner = true } = options;

    Swal.fire({
        title: undefined,
        html: renderPipelineHtml({
            icon: '⏳',
            title: escapeHtml(title),
            subtitle: escapeHtml(subtitle),
            showSpinner,
            variant: 'dark',
        }),
        showConfirmButton: false,
        allowOutsideClick: false,
        background: '#0a1224',
        color: '#e2e8f0',
        backdrop: 'rgba(0,0,0,0.45)',
        customClass: {
            popup: 'progress-dark-popup'
        },
        didOpen: () => {
            const popup = Swal.getPopup();
            if (popup) {
                popup.style.overflow = 'hidden';
                popup.style.maxHeight = 'none';
            }
        }
    });
};

// Non-blocking progress notification: floats in the top-right, no backdrop,
// the user can keep using the app while the pipeline runs in the background.
// `bodyHtml` is optional pre-rendered HTML for richer content (e.g. CE list).
//
// Dark variant — matches the navy-glass language of the rest of the app
// after the dark redesign. The toast is a deep navy card with an indigo
// gradient spinner tile and light text.
const showProgressToast = (title, subtitle, bodyHtml) => {
    const safeBody = bodyHtml
        ? `<div style="margin-top:8px;color:#94a3b8;font-size:12px;line-height:1.45;">${bodyHtml}</div>`
        : '';
    Swal.fire({
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        background: 'rgba(15, 23, 42, 0.92)',
        color: '#e2e8f0',
        html: `
            <div style="display:flex;align-items:flex-start;gap:12px;text-align:left;font-family:'Plus Jakarta Sans',sans-serif;">
                <div style="width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#818cf8 0%,#3b82f6 100%);box-shadow:0 4px 14px -2px rgba(99,102,241,0.55);flex-shrink:0;">
                    <div class="swal2-loader" style="width:18px;height:18px;border-width:2.5px;border-color:rgba(255,255,255,0.35) transparent rgba(255,255,255,0.95) transparent;display:inline-block;border-style:solid;border-radius:50%;animation:swal2-rotate-loading 1.4s linear infinite;margin:0;"></div>
                </div>
                <div style="min-width:0;flex:1;">
                    <div style="font-weight:700;font-size:13px;color:#f1f5f9;letter-spacing:-0.01em;">${escapeHtml(title)}</div>
                    <div style="font-size:12px;color:#94a3b8;margin-top:3px;line-height:1.45;">${escapeHtml(subtitle)}</div>
                    ${safeBody}
                </div>
            </div>
        `,
        customClass: { popup: 'progress-glass-popup' },
        didOpen: () => {
            const popup = Swal.getPopup();
            if (popup) {
                popup.style.maxWidth = '420px';
                popup.style.padding = '14px 16px';
                popup.style.borderRadius = '14px';
                popup.style.border = '1px solid rgba(148, 163, 184, 0.18)';
                popup.style.backdropFilter = 'blur(14px)';
                popup.style.boxShadow = '0 16px 36px -8px rgba(2, 6, 23, 0.55), 0 4px 12px -4px rgba(99, 102, 241, 0.30)';
            }
        }
    });
};

/**
 * Render the name-conflict resolution modal and return the user's decision.
 *
 * Used by the AI pipeline early-detection layer: when the AI proposes a name
 * that's already in the public registry, we surface this BEFORE the user
 * invests minutes in training-data generation that would fail at publish.
 *
 * The user picks one of three actions:
 *   - 'bookmark': add the existing record to bookmarks, abandon the new draft
 *   - 'rename':   pick a different name and proceed with the new local draft
 *   - 'cancel':   abandon the new draft entirely
 *
 * Returns an object describing the choice:
 *   { action: 'bookmark', existingPublicId, existingLocalId }
 *   { action: 'rename',   newName }
 *   { action: 'cancel' }
 *
 * `kind` is "rule" or "CE" (used for copy in the modal). `conflict` is the
 * NameConflict shape from the backend response: { kind, name, existing_public_id, existing_summary }.
 */
const showNameConflictModal = async ({ kind, conflict }) => {
    const summary = conflict.existing_summary || {};
    const existingName = conflict.name;
    const existingPid = conflict.existing_public_id;
    const existingLocalId = summary.local_id;

    // Build a small preview of the existing record so the user can see what
    // they would be bookmarking / what to differentiate their version from.
    const previewBits = [];
    if (kind === 'Rule') {
        if (summary.predicate) previewBits.push(`<p style="margin:6px 0;"><b>Predicate:</b><br/><code style="font-size:0.85em;background:#f3f4f6;padding:6px;border-radius:4px;display:block;word-break:break-word;">${escapeHtml(summary.predicate)}</code></p>`);
        if (summary.description) previewBits.push(`<p style="margin:6px 0;"><b>Description:</b> ${escapeHtml(summary.description)}</p>`);
    } else {
        if (summary.definition) previewBits.push(`<p style="margin:6px 0;"><b>Definition:</b> ${escapeHtml(summary.definition)}</p>`);
        if (summary.category) previewBits.push(`<p style="margin:6px 0;"><b>Category:</b> ${escapeHtml(summary.category)}</p>`);
    }
    if (summary.categories && summary.categories.length) {
        previewBits.push(`<p style="margin:6px 0;"><b>Categories:</b> ${escapeJoined(summary.categories)}</p>`);
    }
    const previewHtml = previewBits.join('') || '<p style="color:#6b7280;"><em>No additional details available — sync your library to see the full record.</em></p>';

    // Top-level "what do you want to do" prompt. SweetAlert can't render
    // three custom buttons natively, so we use the showDenyButton +
    // showCancelButton pattern: confirm = bookmark, deny = rename, cancel = abandon.
    //
    // No `title:` on the Swal config — PipelineModal already renders one
    // in the html body; passing both stacks duplicate titles. Same goes
    // for status icon. Buttons use `buttonsStyling: false` + customClass
    // so they pick up the indigo→blue / outline / ghost styles defined
    // in App.css instead of Swal's stock heavy color blocks.
    const topResult = await Swal.fire({
        html: renderPipelineHtml({
            icon: '⚠️',
            title: `${kind} "${escapeHtml(existingName)}" already exists`,
            subtitle: `Someone has already published a ${kind.toLowerCase()} with this name. You have three options.`,
            variant: 'dark',
            showSpinner: false,
            children: h(
                'div',
                { style: { textAlign: 'left' } },
                h('div', {
                    style: { padding: '10px', background: '#f9fafb', borderRadius: '6px', borderLeft: '4px solid #f59e0b' },
                    dangerouslySetInnerHTML: { __html: previewHtml },
                }),
            ),
        }),
        showCancelButton: true,
        showDenyButton: true,
        confirmButtonText: '📑 Bookmark existing',
        denyButtonText: '✏️ Use a different name',
        cancelButtonText: 'Cancel',
        buttonsStyling: false,
        customClass: {
            popup: 'gavel-publish-conflict-popup',
            confirmButton: 'gavel-publish-confirm-btn',
            denyButton: 'gavel-publish-deny-btn',
            cancelButton: 'gavel-publish-cancel-btn',
        },
        reverseButtons: false,
        width: '650px',
    });

    if (topResult.isConfirmed) {
        return { action: 'bookmark', existingPublicId: existingPid, existingLocalId };
    }
    if (topResult.isDenied) {
        // Prompt for the new name. We loop until they enter a name that
        // either differs from the conflict OR they cancel out.
        // eslint-disable-next-line no-constant-condition
        while (true) {
            const renameResult = await Swal.fire({
                title: 'Choose a new name',
                input: 'text',
                inputLabel: `Pick a name different from "${existingName}".`,
                inputValue: '',
                inputPlaceholder: 'e.g. ' + existingName + '_v2',
                showCancelButton: true,
                confirmButtonText: 'Check & continue',
                inputValidator: (value) => {
                    if (!value || !value.trim()) return 'A name is required';
                    if (value.trim() === existingName) return 'That is the conflicting name; pick something different';
                    return undefined;
                },
            });
            if (!renameResult.isConfirmed) {
                return { action: 'cancel' };
            }
            const candidate = renameResult.value.trim();
            // Live-check against the registry name index. If it's also taken,
            // loop and ask again.
            try {
                const probe = await checkLibraryName({ kind: kind.toLowerCase(), name: candidate });
                if (probe.data?.exists) {
                    await showAlertDialog({
                        title: 'Also taken',
                        message: `"${candidate}" is already in the library. Try another name.`,
                        variant: 'warning',
                    });
                    continue;
                }
            } catch (probeErr) {
                // Probe failed — let them proceed; the publish-time check
                // will catch any remaining conflict.
                console.warn('check-name failed; continuing:', probeErr);
            }
            return { action: 'rename', newName: candidate };
        }
    }
    return { action: 'cancel' };
};


/**
 * Render a Swal that explains the outcome of a publish attempt.
 *
 * The backend's publish service has four terminal states (services/hf_publish.py):
 *   - success: registry now has the record; show a green confirmation.
 *   - conflict: the name is already taken by an existing published record.
 *               Local draft is kept so the user can rename or fork.
 *   - race: another user pushed during ours. Local draft is kept; user retries.
 *   - error: hard failure (network, validation, etc). Backend has already
 *            removed the local draft per the "if something goes wrong we delete it"
 *            contract; user has to redo the AI generation.
 *
 * `kind` is "rule" or "CE" — used for the user-facing copy.
 * `extraSubtitle` is appended to the success-state subtitle for context
 * (e.g. listing how many training-data samples were generated).
 */
const showPublishResultModal = async ({ kind, result, extraSubtitle = '' }) => {
    const status = result?.status;
    const name = result?.name || result?.public_id || '';

    // Common Swal config for all publish-result variants. We deliberately
    // DO NOT pass `icon: 'success'/'error'/...` — the PipelineModal HTML
    // already renders an emoji-in-rounded-tile, and Swal's built-in icon
    // would stack a second one above it (looked redundant + cramped).
    // Confirm button uses the indigo→blue gradient via customClass so it
    // matches the rest of the modernized app's primary buttons; status
    // signal still comes through via PipelineModal's icon tile + copy.
    const baseConfig = {
        showConfirmButton: true,
        confirmButtonText: 'Got it',
        buttonsStyling: false,
        customClass: {
            popup: 'gavel-publish-result-popup',
            confirmButton: 'gavel-publish-confirm-btn',
        },
        width: '520px',
    };

    if (status === 'success') {
        await Swal.fire({
            ...baseConfig,
            html: renderPipelineHtml({
                icon: '✅',
                title: `${kind} published to library`,
                subtitle: `"${escapeHtml(name)}" is now in the public registry. ${extraSubtitle}`.trim(),
                variant: 'dark',
                showSpinner: false,
            }),
        });
        return;
    }

    if (status === 'conflict') {
        const cw = result.conflict_with || {};
        await Swal.fire({
            ...baseConfig,
            html: renderPipelineHtml({
                icon: '⚠️',
                title: 'Already in the library',
                subtitle: `A ${escapeHtml(cw.type || kind.toLowerCase())} called "${escapeHtml(cw.name)}" already exists. Either rename your draft or use Fork & Edit on the existing one.`,
                variant: 'dark',
                showSpinner: false,
            }),
        });
        return;
    }

    if (status === 'race') {
        await Swal.fire({
            ...baseConfig,
            html: renderPipelineHtml({
                icon: '🔄',
                title: 'Another user just published',
                subtitle: 'The library was updated by someone else during your push. Your draft is preserved — please retry to publish.',
                variant: 'dark',
                showSpinner: false,
            }),
        });
        return;
    }

    // error or unknown
    await Swal.fire({
        ...baseConfig,
        html: renderPipelineHtml({
            icon: '⛔',
            title: `${kind} publish failed`,
            subtitle: `${escapeHtml(result?.error || 'Unknown error')}. The local draft has been removed; please redo the AI generation.`,
            variant: 'dark',
            showSpinner: false,
        }),
    });
};


/**
 * Publish an already-existing local-draft rule to the public registry.
 *
 * Triggered by the "Publish to library" button on RuleCard (for rules with
 * is_local_draft === true). The flow:
 *
 *   1. Probe the registry name index for `rule.custom_name`. If it's taken,
 *      surface the same name-conflict resolver the AI pipeline uses
 *      (bookmark existing / rename / cancel).
 *      - On "bookmark": discard our local draft, force-sync, bookmark the
 *        existing public record.
 *      - On "rename": persist the new name to the local rule row, then
 *        proceed to push.
 *   2. Push to HuggingFace via /library/publish/rule/{rule_id}, which
 *      handles the sync-first + dedup + race-checked commit atomically
 *      (services/hf_publish.py).
 *   3. Show the unified outcome modal.
 *
 * `rule` is the RuleCard's rule object — must have `source_rule_id` (the
 * underlying rules-table id) and `custom_name`. `userId` is needed for
 * bookmark routing on the conflict path. `refreshData` is the caller's
 * data-reload callback.
 */
export const publishDraftRule = async (rule, userId, refreshData, _attempt = 0) => {
    // Recursion guard: this function re-invokes itself after adopt/rename-CE to
    // re-run publish. If the backend keeps returning the SAME conflict (e.g. the
    // adopt silently no-ops), bail instead of recursing into a stack overflow.
    if (_attempt > 3) {
        await showAlertDialog({
            title: 'Could not resolve the conflict',
            message: 'The publish kept hitting the same element conflict after several attempts. Please resolve it manually (rename the element or re-sync) and try again.',
            variant: 'error',
        });
        if (typeof refreshData === 'function') refreshData();
        return;
    }
    const ruleId = rule.source_rule_id || rule.rule_id;
    const ruleName = rule.custom_name || rule.name;
    if (!ruleId) {
        await showAlertDialog({
            title: 'Cannot publish yet',
            message: 'This rule was assembled from bookmarks and has no underlying registry record. Use the AI pipeline or "Create public rule" flow to publish.',
            variant: 'info',
        });
        return;
    }

    // Force a fresh library sync BEFORE the name probe. Even with live
    // push, a friend on another machine could have published a rule with
    // this name in the brief window before our stream applied it.
    // Running force-sync first means our probe sees the latest registry
    // state, so we surface the conflict modal instead of letting the
    // publish call eat a CONFLICT response.
    try {
        await api.get('/library/sync', { params: { force: true } });
    } catch (preSyncErr) {
        // Best-effort: a failed pre-sync just means we fall back to
        // the existing post-publish dedup. Don't block on it.
        console.warn('Pre-publish library sync failed; continuing:', preSyncErr);
    }

    // Pre-check the name. The publish endpoint checks again (and the HF
    // commit is parent_commit-protected), but probing here lets us hand the
    // user the rich conflict-resolution modal instead of a raw error toast.
    let renamed = false;
    try {
        const probe = await checkLibraryName({ kind: 'rule', name: ruleName });
        if (probe.data?.exists) {
            const choice = await showNameConflictModal({
                kind: 'Rule',
                conflict: {
                    kind: 'rule',
                    name: ruleName,
                    existing_public_id: probe.data.public_id,
                    existing_summary: probe.data.summary,
                },
            });

            if (choice.action === 'cancel') {
                return;
            }

            if (choice.action === 'bookmark') {
                // Drop our local draft so the next sync can pull the
                // registry version unimpeded (sync skips overwrites when a
                // same-named draft exists).
                try {
                    await api.post('/ai/discard-pipeline-resources', {
                        ce_ids: [],
                        rule_id: ruleId,
                    });
                } catch (discardErr) {
                    console.error('Discard before bookmark failed:', discardErr);
                }
                try {
                    await api.get('/library/sync', { params: { force: true } });
                } catch (syncErr) {
                    console.error('Sync before bookmark failed:', syncErr);
                }
                try {
                    const recResp = await api.get(
                        `/library/record/rule/${encodeURIComponent(choice.existingPublicId)}`
                    );
                    const localId = recResp.data?.summary?.local_id;
                    if (!localId) throw new Error('Rule not found locally after sync');
                    if (userId) await addRuleBookmark(userId, localId);
                    await showAlertDialog({
                        title: 'Saved',
                        message: `"${ruleName}" is now in your bookmarks.`,
                        variant: 'success',
                    });
                } catch (bookmarkErr) {
                    console.error('Bookmark failed:', bookmarkErr);
                    await showAlertDialog({
                        title: 'Could not bookmark',
                        message: bookmarkErr.message || 'Try syncing manually and bookmarking from the library page.',
                        variant: 'error',
                    });
                }
                if (typeof refreshData === 'function') refreshData();
                return;
            }

            if (choice.action === 'rename') {
                try {
                    await api.post('/ai/rename-rule', {
                        rule_id: ruleId,
                        new_name: choice.newName,
                    });
                    renamed = true;
                } catch (renameErr) {
                    console.error('Rename failed:', renameErr);
                    await showAlertDialog({
                        title: 'Rename failed',
                        message: renameErr.response?.data?.detail || renameErr.message || 'Could not save the new name.',
                        variant: 'error',
                    });
                    return;
                }
            }
        }
    } catch (probeErr) {
        // Probe failed — let publish-time check catch any leftover conflict.
        console.warn('Rule name probe failed; continuing:', probeErr);
    }

    showProgressToast(
        'Publishing to library',
        'Pushing the rule and any new CEs to the public registry…'
    );
    let publishResult;
    try {
        const publishResp = await publishRule(ruleId);
        publishResult = publishResp.data;
        // Auto-save your own freshly published rule to your bookmarks. Now that
        // it has a public_id it CAN be bookmarked (drafts can't — bookmarks are
        // keyed by the public id), so it lands in "My Bookmarked Rules" and is
        // manageable (remove / re-add) like any public rule. Best-effort: the
        // publish already succeeded regardless.
        if (userId && publishResult?.status === 'success') {
            try { await addRuleBookmark(userId, ruleId); }
            catch (e) { console.error('auto-bookmark after publish failed:', e); }
        }
    } catch (err) {
        publishResult = {
            status: 'error',
            error: err.response?.data?.detail || err.message || 'Network error',
            name: ruleName,
        };
    }

    // A draft CE inside this rule name-clashes with a public CE. There's no rule
    // editor to fix it by hand, so we offer two one-click resolutions:
    //   • Use shared — replace the draft with the public CE in place, then publish.
    //   • Rename mine — give the draft a new name, then publish it as a new CE.
    const cw0 = publishResult?.conflict_with;
    if (publishResult?.status === 'conflict' && cw0?.type === 'ce'
        && cw0.public_id && cw0.local_ce_id) {
        Swal.close();
        const ceName = escapeHtml(cw0.name);
        const choice = await Swal.fire({
            showDenyButton: true,
            showCancelButton: true,
            confirmButtonText: 'Use shared CE & publish',
            denyButtonText: 'Rename my CE & publish',
            cancelButtonText: 'Cancel',
            buttonsStyling: false,
            customClass: {
                popup: 'gavel-publish-result-popup',
                confirmButton: 'gavel-publish-confirm-btn',
                denyButton: 'gavel-publish-confirm-btn',
                cancelButton: 'gavel-publish-confirm-btn',
            },
            width: '600px',
            html: renderPipelineHtml({
                icon: '🔗',
                title: 'A building block with this name already exists',
                subtitle: `Your rule uses a cognitive element “${ceName}”, but one with that name is `
                    + `already shared in the public library. Pick one: `
                    + `“Use shared CE” replaces your copy of “${ceName}” with the shared one and publishes your rule using it — no duplicate is created. `
                    + `“Rename my CE” keeps your version under a new name and publishes your rule with that new element instead.`,
                variant: 'dark',
                showSpinner: false,
            }),
        });

        // --- Use the shared CE (adopt in place), then re-publish. ---
        if (choice.isConfirmed) {
            try {
                showProgressToast('Using shared element', `Replacing “${cw0.name}” with the public version…`);
                await adoptPublicCE(cw0.local_ce_id, cw0.public_id);
                Swal.close();
            } catch (adoptErr) {
                Swal.close();
                await showAlertDialog({
                    title: 'Could not use the shared element',
                    message: adoptErr.response?.data?.detail || adoptErr.message || 'Failed to adopt the shared element.',
                    variant: 'error',
                });
                if (typeof refreshData === 'function') refreshData();
                return;
            }
            // CE is now public → re-run publish (handles any further clashes too).
            return publishDraftRule(rule, userId, refreshData, _attempt + 1);
        }

        // --- Rename my CE, then re-publish it as a new element. ---
        if (choice.isDenied) {
            const renameRes = await Swal.fire({
                title: `Rename your “${cw0.name}”`,
                input: 'text',
                inputLabel: `Pick a name different from “${cw0.name}”.`,
                inputValue: '',
                inputPlaceholder: cw0.name + '_v2',
                showCancelButton: true,
                confirmButtonText: 'Rename & publish',
                buttonsStyling: false,
                customClass: {
                    popup: 'gavel-publish-result-popup',
                    confirmButton: 'gavel-publish-confirm-btn',
                    cancelButton: 'gavel-publish-confirm-btn',
                },
                inputValidator: (value) => {
                    if (!value || !value.trim()) return 'A new name is required';
                    if (value.trim() === cw0.name) return 'That is the conflicting name; pick something different';
                    return undefined;
                },
            });
            if (!renameRes.isConfirmed) {
                if (typeof refreshData === 'function') refreshData();
                return;
            }
            const newName = renameRes.value.trim();
            try {
                showProgressToast('Renaming', `Renaming “${cw0.name}” to “${newName}”…`);
                await api.post('/ai/rename-ce', { ce_id: cw0.local_ce_id, new_name: newName });
                Swal.close();
            } catch (renameErr) {
                Swal.close();
                await showAlertDialog({
                    title: 'Rename failed',
                    message: renameErr.response?.data?.detail || renameErr.message || 'Could not rename the element.',
                    variant: 'error',
                });
                if (typeof refreshData === 'function') refreshData();
                return;
            }
            // CE now has a unique name → re-run publish (it goes up as a new CE).
            return publishDraftRule(rule, userId, refreshData, _attempt + 1);
        }

        // Cancelled — leave the draft as-is.
        if (typeof refreshData === 'function') refreshData();
        return;
    }

    Swal.close();
    await showPublishResultModal({
        kind: 'Rule',
        result: publishResult,
    });
    if (typeof refreshData === 'function') refreshData();
    // `renamed` is local state; if we want to surface "we renamed it for
    // you" copy in the UI later, this is the hook.
    void renamed;
};

/**
 * Publish an already-existing local-draft CE to the public registry.
 *
 * Triggered by the "Publish" button on CognitiveElementCard for CEs with
 * is_local_draft === true. Mirrors publishDraftRule:
 *   1. Probe the registry name index. If taken, surface the conflict modal
 *      (bookmark / rename / cancel).
 *   2. Push to /library/publish/ce/{ce_id}.
 *   3. Show the unified outcome modal.
 *
 * `ce` must have `ce_id` and `name`. `userId` is needed for bookmark routing
 * on the conflict path. `refreshData` reloads the caller's CE list.
 */
export const publishDraftCE = async (ce, userId, refreshData) => {
    const ceId = ce.ce_id;
    const ceName = ce.name;
    if (!ceId) {
        await showAlertDialog({
            title: 'Cannot publish yet',
            message: 'This CE is missing its identifier.',
            variant: 'info',
        });
        return;
    }

    // Force-sync before the name probe (same reasoning as publishDraftRule):
    // catch any same-named CE another user pushed since our last poll
    // tick, so the conflict modal fires here rather than at publish time.
    try {
        await api.get('/library/sync', { params: { force: true } });
    } catch (preSyncErr) {
        console.warn('Pre-publish library sync failed; continuing:', preSyncErr);
    }

    try {
        const probe = await checkLibraryName({ kind: 'ce', name: ceName });
        if (probe.data?.exists) {
            const choice = await showNameConflictModal({
                kind: 'CE',
                conflict: {
                    kind: 'ce',
                    name: ceName,
                    existing_public_id: probe.data.public_id,
                    existing_summary: probe.data.summary,
                },
            });

            if (choice.action === 'cancel') {
                return;
            }

            if (choice.action === 'bookmark') {
                // Drop our local draft so the next sync can pull the registry
                // version (sync skips overwrites when there's a same-named draft).
                try {
                    await api.post('/ai/discard-pipeline-resources', {
                        ce_ids: [ceId],
                        rule_id: null,
                    });
                } catch (discardErr) {
                    console.error('Discard before bookmark failed:', discardErr);
                }
                try {
                    await api.get('/library/sync', { params: { force: true } });
                } catch (syncErr) {
                    console.error('Sync before bookmark failed:', syncErr);
                }
                try {
                    const recResp = await api.get(
                        `/library/record/ce/${encodeURIComponent(choice.existingPublicId)}`
                    );
                    const localId = recResp.data?.summary?.local_id;
                    if (!localId) throw new Error('CE not found locally after sync');
                    if (userId) await addCEBookmark(userId, localId);
                    await showAlertDialog({
                        title: 'Saved',
                        message: `"${ceName}" is now in your bookmarks.`,
                        variant: 'success',
                    });
                } catch (bookmarkErr) {
                    console.error('Bookmark failed:', bookmarkErr);
                    await showAlertDialog({
                        title: 'Could not bookmark',
                        message: bookmarkErr.message || 'Try syncing manually.',
                        variant: 'error',
                    });
                }
                if (typeof refreshData === 'function') refreshData();
                return;
            }

            if (choice.action === 'rename') {
                try {
                    await api.post('/ai/rename-ce', {
                        ce_id: ceId,
                        new_name: choice.newName,
                    });
                } catch (renameErr) {
                    console.error('Rename failed:', renameErr);
                    await showAlertDialog({
                        title: 'Rename failed',
                        message: renameErr.response?.data?.detail || renameErr.message || 'Could not save the new name.',
                        variant: 'error',
                    });
                    return;
                }
            }
        }
    } catch (probeErr) {
        console.warn('CE name probe failed; continuing:', probeErr);
    }

    showProgressToast(
        'Publishing CE to library',
        'Pushing the cognitive element and its dataset to the public registry…'
    );
    let publishResult;
    try {
        const publishResp = await publishCE(ceId);
        publishResult = publishResp.data;
        // Auto-save your own freshly published CE to your bookmarks (same
        // reasoning as rules: a public_id now exists, so it's bookmarkable and
        // manageable like any public CE). Best-effort.
        if (userId && publishResult?.status === 'success') {
            try { await addCEBookmark(userId, ceId); }
            catch (e) { console.error('auto-bookmark after publish failed:', e); }
        }
    } catch (err) {
        publishResult = {
            status: 'error',
            error: err.response?.data?.detail || err.message || 'Network error',
            name: ceName,
        };
    }
    Swal.close();
    await showPublishResultModal({
        kind: 'CE',
        result: publishResult,
    });
    if (typeof refreshData === 'function') refreshData();
};


/**
 * Share a private rule set (a model-less guardrail / classifier) to the
 * community. Unlike rules/CEs there is no name-conflict pre-probe modal — a
 * rule set is a thin pointer-collection of ALREADY-published rules, so the
 * backend's only soft failure is "publish the member rules first", which we
 * surface verbatim because it's directly actionable.
 *
 * `classifierId` is the private rule set's id; `name` is its display name;
 * `refreshData` reloads the caller's list.
 */
export const publishDraftRuleSet = async (classifierId, name, refreshData) => {
    if (!classifierId) {
        await showAlertDialog({ title: 'Cannot share yet', message: 'This rule set is missing its identifier.', variant: 'info' });
        return;
    }
    showProgressToast('Sharing rule set', 'Publishing your rule set to the community…');
    let result;
    try {
        const resp = await publishRuleSet(classifierId);
        result = resp.data;
    } catch (err) {
        result = { status: 'error', error: err.response?.data?.detail || err.message || 'Network error', name };
    }
    Swal.close();

    const status = result?.status;
    if (status === 'success') {
        await showAlertDialog({
            title: 'Rule set shared',
            message: `"${result.name || name}" is now in the community library.`,
            variant: 'success',
        });
    } else if (status === 'conflict') {
        const cw = result.conflict_with || {};
        await showAlertDialog({
            title: 'Name already taken',
            message: `A public rule set called "${cw.name || name}" already exists. Rename your rule set and try again.`,
            variant: 'warning',
        });
    } else if (status === 'race') {
        await showAlertDialog({
            title: 'Another user just published',
            message: 'The library changed during your push. Nothing was shared — please try again.',
            variant: 'info',
        });
    } else {
        // error — surface the backend message (e.g. the members-first gate).
        await showAlertDialog({
            title: 'Could not share rule set',
            message: result?.error || 'Unknown error.',
            variant: 'error',
        });
    }
    if (typeof refreshData === 'function') refreshData();
};
