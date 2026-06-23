// Behavior tests for the BuildRuleFromCEs wizard body.
//
// Now rendered as the BODY of BuildRuleFromCEsModal (no longer a routed page):
// it composes a rule from the user's bookmarked Cognitive Elements across five
// steps (Pick CEs -> Learn Roles -> Assign -> Name -> Test & Calibration), then
// creates a provisional draft and (via the background tray) finalizes it. The
// page chrome is gone — `onClose` closes the modal instead of navigating.
//
// Strategy:
//   * mock '../api' so nothing hits the network — each export returns benign data
//   * mock showAlertDialog (confirmDialog) so validation prompts are observable
//     and never open a real Swal popup (which the console.error fail-fast hates)
//   * stub RuleDefaultsStep so step 5 is deterministic and exposes onDone/finalize

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TaskTrayProvider } from '../../src/contexts/TaskTrayContext';

// --- mock the API ---
vi.mock('../../src/api', () => ({
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    getAllCategories: vi.fn(() => Promise.resolve({ data: [] })),
    createDraftRuleFromBookmarks: vi.fn(() => Promise.resolve({ data: { rule_id: 99 } })),
    finalizeRule: vi.fn(() => Promise.resolve({ data: {} })),
    discardUnreadyRule: vi.fn(() => Promise.resolve({ data: {} })),
}));

// --- mock showAlertDialog so validation prompts are observable + silent ---
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: vi.fn(() => Promise.resolve()),
}));

// --- stub the final step so step 5 is deterministic ---
// The real component calls onDone immediately (navigate away) and finalize only
// AFTER its background test-set generation finishes. The stub exposes both as
// separate buttons so tests can drive each independently.
vi.mock('../../src/components/RuleDefaults/RuleDefaultsStep', () => ({
    default: ({ ruleId, onDone, finalize }) => (
        <div data-testid="rule-defaults-step">
            <span data-testid="rds-rule-id">{String(ruleId)}</span>
            <button onClick={onDone}>finish-defaults</button>
            <button onClick={finalize}>finalize-defaults</button>
        </div>
    ),
}));

import {
    getCEBookmarks,
    getAllCategories,
    createDraftRuleFromBookmarks,
    finalizeRule,
    discardUnreadyRule,
} from '../../src/api';
import { showAlertDialog } from '../../src/components/ConfirmDialog/confirmDialog';
import BuildRuleFromCEs from '../../src/pages/BuildRuleFromCEs';

const USER = { user_id: 42, email: 'a@b.c' };

const BOOKMARKS = [
    { ce_id: 1, name: 'Alpha CE', category: 'Safety' },
    { ce_id: 2, name: 'Beta CE', category: 'Privacy' },
    { ce_id: 3, name: 'Gamma CE', category: 'Ethics' },
];

const setUser = (u = USER) => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify(u));
};

// Module-scoped so every test (and the inline renders below) shares one spy.
let onCloseMock = vi.fn();
const renderPage = () => render(
    <TaskTrayProvider>
        <BuildRuleFromCEs onClose={onCloseMock} />
    </TaskTrayProvider>,
);

// Wait for the initial load() effect to resolve (loading -> false).
const waitLoaded = async () =>
    waitFor(() => expect(screen.queryByText(/Loading your bookmarked CEs/i)).not.toBeInTheDocument());

// Select a CE checkbox by visible name on step 1.
const toggleCe = (name) => {
    const label = screen.getByText(name).closest('label');
    fireEvent.click(within(label).getByRole('checkbox'));
};

// On the Assign step (step 3) each selected CE renders as a row containing its
// name and a segmented group of role <button>s (Necessary / Fallback / Helpful).
// The name lives in a <div> whose parent is the row. Find that row by CE name.
const ceRow = (name) =>
    screen.getAllByText(name).find((el) => el.tagName === 'DIV').closest('div').parentElement;

// Click a role button (by visible label) within a given CE's row.
const setRole = (ceName, roleLabel) =>
    fireEvent.click(within(ceRow(ceName)).getByRole('button', { name: roleLabel }));


describe('BuildRuleFromCEs', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        onCloseMock = vi.fn();
        setUser();
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
        getAllCategories.mockResolvedValue({ data: [] });
        createDraftRuleFromBookmarks.mockResolvedValue({ data: { rule_id: 99 } });
        finalizeRule.mockResolvedValue({ data: {} });
        discardUnreadyRule.mockResolvedValue({ data: {} });
    });

    // ---- mount / load -------------------------------------------------------

    it('closes the modal when no user is in session storage', async () => {
        sessionStorage.clear();
        renderPage();
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
        // Never fetched bookmarks for a missing user.
        expect(getCEBookmarks).not.toHaveBeenCalled();
    });

    it('shows the loading state before data resolves, then clears it', async () => {
        renderPage();
        expect(screen.getByText(/Loading your bookmarked CEs/i)).toBeInTheDocument();
        await waitLoaded();
    });

    it('renders the intro copy and step indicator labels', async () => {
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/Compose a rule from your bookmarked Cognitive Elements/i)).toBeInTheDocument();
        ['Pick CEs', 'Learn Roles', 'Assign', 'Name', 'Test & Calibration']
            .forEach((l) => expect(screen.getByText(l)).toBeInTheDocument());
    });

    it('fetches bookmarks with the user id and categories on mount', async () => {
        renderPage();
        await waitLoaded();
        expect(getCEBookmarks).toHaveBeenCalledWith(42);
        expect(getAllCategories).toHaveBeenCalled();
    });

    it('shows the empty state when there are no bookmarked CEs', async () => {
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/No bookmarked CEs/i)).toBeInTheDocument();
        expect(screen.getByText(/0 selected/i)).toBeInTheDocument();
    });

    it('renders each bookmarked CE with its name and category', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        expect(screen.getByText('Alpha CE')).toBeInTheDocument();
        expect(screen.getByText('Beta CE')).toBeInTheDocument();
        expect(screen.getByText('Safety')).toBeInTheDocument();
        expect(screen.getByText('Privacy')).toBeInTheDocument();
    });

    it('falls back to an empty list when getCEBookmarks rejects', async () => {
        getCEBookmarks.mockRejectedValue(new Error('boom'));
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/No bookmarked CEs/i)).toBeInTheDocument();
    });

    it('handles a categories response that is not an array', async () => {
        // catRes.data not an array -> availableCategories stays empty (no crash).
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: { unexpected: true } });
        renderPage();
        await waitLoaded();
        // Proceed to the Name step and confirm the "no categories" hint shows.
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));            // -> step 2
        fireEvent.click(screen.getByText('Next'));            // -> step 3
        fireEvent.click(screen.getByText('Next'));            // -> step 4
        expect(screen.getByText(/No categories available yet/i)).toBeInTheDocument();
    });

    // ---- step 1: pick CEs ---------------------------------------------------

    it('warns and stays on step 1 when Next is clicked with nothing selected', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        fireEvent.click(screen.getByText('Next'));
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Select CEs' }),
        );
        // Still on step 1 (the "selected" counter is a step-1-only element).
        expect(screen.getByText(/0 selected/i)).toBeInTheDocument();
    });

    it('updates the selected counter as CEs are checked and unchecked', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        expect(screen.getByText(/1 selected/i)).toBeInTheDocument();
        toggleCe('Beta CE');
        expect(screen.getByText(/2 selected/i)).toBeInTheDocument();
        toggleCe('Alpha CE'); // uncheck
        expect(screen.getByText(/1 selected/i)).toBeInTheDocument();
    });

    // ---- step 2 -> 3: learn roles / assign ---------------------------------

    it('advances through Learn Roles into Assign and lists selected CEs with role selects', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // -> step 2
        expect(screen.getByText(/How roles shape the predicate/i)).toBeInTheDocument();
        fireEvent.click(screen.getByText('Next'));   // -> step 3
        expect(screen.getByText(/Assign a role to each CE/i)).toBeInTheDocument();
        // Each selected CE shows a segmented role control with the three
        // role buttons (Necessary / Fallback / Helpful).
        ['Alpha CE', 'Beta CE'].forEach((name) => {
            const row = ceRow(name);
            expect(within(row).getByRole('button', { name: 'Necessary' })).toBeInTheDocument();
            expect(within(row).getByRole('button', { name: 'Any of' })).toBeInTheDocument();
            expect(within(row).getByRole('button', { name: 'Supporting' })).toBeInTheDocument();
        });
    });

    it('Back from Learn Roles returns to Pick CEs', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        fireEvent.click(screen.getByText('Next'));   // -> step 2
        fireEvent.click(screen.getByText('Back'));   // -> step 1
        expect(screen.getByText(/Select the cognitive elements/i)).toBeInTheDocument();
    });

    it('changing a role to Fallback reveals the OR Group number input', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // step 2
        fireEvent.click(screen.getByText('Next'));   // step 3

        expect(screen.queryByText(/OR Group/i)).not.toBeInTheDocument();
        setRole('Alpha CE', 'Any of');
        expect(screen.getByText(/OR Group/i)).toBeInTheDocument();

        // The number input defaults to 1 and accepts an edit.
        const num = screen.getByRole('spinbutton');
        expect(num).toHaveValue(1);
        fireEvent.change(num, { target: { value: '3' } });
        expect(num).toHaveValue(3);
    });

    it('selecting Sufficient does not show the OR Group input', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // step 2
        fireEvent.click(screen.getByText('Next'));   // step 3
        setRole('Alpha CE', 'Supporting');
        expect(screen.queryByText(/OR Group/i)).not.toBeInTheDocument();
    });

    // ---- step 4: name + categories -----------------------------------------

    const gotoStep4 = async ({ categories = ['Safety & Harm', 'Privacy'] } = {}) => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: categories });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // -> 2
        fireEvent.click(screen.getByText('Next'));   // -> 3
        fireEvent.click(screen.getByText('Next'));   // -> 4
    };

    it('renders the name input and category chips on the Name step', async () => {
        await gotoStep4();
        expect(screen.getByPlaceholderText(/phishing_content_creation/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Safety & Harm' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Privacy' })).toBeInTheDocument();
    });

    it('keeps Create Rule disabled until a name and a category are provided', async () => {
        await gotoStep4();
        const createBtn = () => screen.getByText('Create Rule').closest('button');
        expect(createBtn()).toBeDisabled();

        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'my_rule' } });
        expect(createBtn()).toBeDisabled(); // name alone is not enough

        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        expect(createBtn()).not.toBeDisabled();
    });

    it('toggling a category chip on and off flips the Create button enablement', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        const chip = screen.getByRole('button', { name: 'Privacy' });
        fireEvent.click(chip);
        expect(screen.getByText('Create Rule').closest('button')).not.toBeDisabled();
        fireEvent.click(chip); // deselect
        expect(screen.getByText('Create Rule').closest('button')).toBeDisabled();
    });

    it('Back from the Name step returns to Assign', async () => {
        await gotoStep4();
        fireEvent.click(screen.getByText('Back'));
        expect(screen.getByText(/Assign a role to each CE/i)).toBeInTheDocument();
    });

    it('dedupes and sorts category names from the API', async () => {
        // normalizeCategoryValue strips brackets; duplicates collapse; output sorted.
        await gotoStep4({ categories: ['Zebra', '[Apple]', 'Apple', 'Mango'] });
        const chips = screen.getAllByRole('button')
            .map((b) => b.textContent)
            .filter((t) => ['Apple', 'Mango', 'Zebra'].includes(t));
        expect(chips).toEqual(['Apple', 'Mango', 'Zebra']);
    });

    // ---- handleCreate -------------------------------------------------------

    it('creates the draft rule with trimmed name, ce links, and categories then advances to step 5', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: '  my_rule  ' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(createDraftRuleFromBookmarks).toHaveBeenCalled());
        const [name, ceLinks, categories] = createDraftRuleFromBookmarks.mock.calls[0];
        expect(name).toBe('my_rule');
        expect(categories).toEqual(['Safety & Harm']);
        expect(ceLinks).toEqual([
            { ce_id: 1, role: 'necessary', fallback_group: 0 },
            { ce_id: 2, role: 'necessary', fallback_group: 0 },
        ]);

        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());
        expect(screen.getByTestId('rds-rule-id')).toHaveTextContent('99');
    });

    it('encodes fallback_group only for fallback CEs in the ce links', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // -> 2
        fireEvent.click(screen.getByText('Next')); // -> 3
        // Make Alpha a fallback in group 2.
        setRole('Alpha CE', 'Any of');
        fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '2' } });
        fireEvent.click(screen.getByText('Next')); // -> 4

        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(createDraftRuleFromBookmarks).toHaveBeenCalled());
        const ceLinks = createDraftRuleFromBookmarks.mock.calls[0][1];
        expect(ceLinks).toContainEqual({ ce_id: 1, role: 'fallback', fallback_group: 2 });
        expect(ceLinks).toContainEqual({ ce_id: 2, role: 'necessary', fallback_group: 0 });
    });

    it('shows an error alert and stays on step 4 when create fails', async () => {
        createDraftRuleFromBookmarks.mockRejectedValue({ response: { data: { detail: 'nope' } } });
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Could not create rule', message: 'nope' }),
        ));
        // Did not advance to step 5.
        expect(screen.queryByTestId('rule-defaults-step')).not.toBeInTheDocument();
    });

    it('shows an error alert when the create response is missing a rule_id', async () => {
        createDraftRuleFromBookmarks.mockResolvedValue({ data: {} });
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Could not create rule' }),
        ));
        expect(screen.queryByTestId('rule-defaults-step')).not.toBeInTheDocument();
    });

    // ---- step 5: finalize ---------------------------------------------------

    it('closes the modal immediately on start, WITHOUT revealing the rule yet', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        // onDone fires immediately so the modal closes — but the rule is NOT
        // finalized (revealed) yet; that waits for the background test set.
        fireEvent.click(screen.getByText('finish-defaults'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
        expect(finalizeRule).not.toHaveBeenCalled();
    });

    it('reveals the rule (finalizeRule with id + ce ids) only when finalize runs after the set is ready', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        // The background job calls finalize() after the test set finishes.
        fireEvent.click(screen.getByText('finalize-defaults'));
        await waitFor(() => expect(finalizeRule).toHaveBeenCalledWith(99, [1, 2]));
    });

    // ---- unmount cleanup ----------------------------------------------------

    it('does NOT discard the rule on unmount after it was committed (finished)', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        const { unmount } = render(
            <TaskTrayProvider>
                <BuildRuleFromCEs onClose={onCloseMock} />
            </TaskTrayProvider>,
        );
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // 2
        fireEvent.click(screen.getByText('Next')); // 3
        fireEvent.click(screen.getByText('Next')); // 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());
        fireEvent.click(screen.getByText('finish-defaults'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());

        discardUnreadyRule.mockClear();
        // Unmounting now must NOT discard — committedRef was set true on finish.
        unmount();
        expect(discardUnreadyRule).not.toHaveBeenCalled();
    });

    it('does not discard anything on unmount when no rule was ever created', async () => {
        const { unmount } = renderPage();
        await waitLoaded();
        unmount();
        expect(discardUnreadyRule).not.toHaveBeenCalled();
    });

    it('discards the provisional rule on unmount (modal closed before kicking off the build)', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        const { unmount } = render(
            <TaskTrayProvider>
                <BuildRuleFromCEs onClose={onCloseMock} />
            </TaskTrayProvider>,
        );
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // 2
        fireEvent.click(screen.getByText('Next')); // 3
        fireEvent.click(screen.getByText('Next')); // 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        unmount();
        await waitFor(() => expect(discardUnreadyRule).toHaveBeenCalledWith(99));
    });
});
