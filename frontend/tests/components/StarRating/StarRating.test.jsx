// Behavior tests for StarRating.
//
// StarRating fetches its own rating summary on mount and renders one of:
//   * nothing            (no asset_public_id)
//   * a loading skeleton (summary not yet resolved)
//   * an interactive 1-5 star strip (authenticated non-author)
//   * a read-only star strip (the artifact's author)
//
// We mock '../../api' so nothing hits the network and drive the three
// rating endpoints (getRatingSummary / rateAsset / withdrawRating) with
// deterministic resolved/rejected values per test.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// --- Mock the api module. Each function is a vi.fn so individual tests
// can override resolved/rejected values. Defaults are benign.
vi.mock('../../../src/api', () => ({
    getRatingSummary: vi.fn(() => Promise.resolve({ data: null })),
    rateAsset: vi.fn(() => Promise.resolve({ data: null })),
    withdrawRating: vi.fn(() => Promise.resolve({ data: null })),
}));

// Stub the star icon so we don't depend on react-icons internals.
vi.mock('react-icons/fi', () => ({
    FiStar: () => <svg data-testid="fi-star" />,
}));

import StarRating from '../../../src/components/StarRating/StarRating';
import { getRatingSummary, rateAsset, withdrawRating } from '../../../src/api';

const setUser = (username = 'alice') => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 1, username }));
};

const summaryFor = (over = {}) => ({
    asset_type: 'rule',
    asset_public_id: 'pub-1',
    rating_count: 0,
    rating_avg: null,
    your_score: null,
    ...over,
});

// Render and wait for the mount-effect's getRatingSummary promise to flush.
const renderResolved = async (props) => {
    let utils;
    await act(async () => {
        utils = render(<StarRating {...props} />);
    });
    return utils;
};

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser('alice');
    // Default: a resolved empty summary so the widget leaves the skeleton.
    getRatingSummary.mockResolvedValue({ data: summaryFor() });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('StarRating — gating', () => {
    it('renders nothing without an asset_public_id', () => {
        const { container } = render(
            <StarRating asset_type="rule" asset_public_id={undefined} author_username="bob" />,
        );
        expect(container.firstChild).toBeNull();
        expect(getRatingSummary).not.toHaveBeenCalled();
    });

    it('shows a loading skeleton before the summary resolves', () => {
        // Pending promise: never resolves during this synchronous assertion.
        getRatingSummary.mockReturnValue(new Promise(() => {}));
        render(<StarRating asset_type="rule" asset_public_id="pub-1" author_username="bob" />);
        expect(screen.getByLabelText(/Loading ratings/i)).toBeInTheDocument();
    });

    it('fetches the summary on mount with the right args', async () => {
        await renderResolved({ asset_type: 'ce', asset_public_id: 'pub-9', author_username: 'bob' });
        expect(getRatingSummary).toHaveBeenCalledWith('ce', 'pub-9');
    });
});

describe('StarRating — non-author interactive rendering', () => {
    it('renders 5 star buttons and the "Rate this" label when unrated', async () => {
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        expect(stars).toHaveLength(5);
        expect(screen.getByText('Rate this')).toBeInTheDocument();
        // Unrated => no filled stars.
        stars.forEach((s) => expect(s.className).not.toMatch(/filled/));
    });

    it('shows "Be the first to rate" when count is 0', async () => {
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByText('Be the first to rate')).toBeInTheDocument();
    });

    it('shows "Your rating" label and fills stars up to your_score', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ your_score: 3, rating_count: 4, rating_avg: 3.5 }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByText('Your rating')).toBeInTheDocument();
        const stars = screen.getAllByRole('radio');
        // First 3 filled, last 2 not.
        expect(stars[0].className).toMatch(/filled/);
        expect(stars[2].className).toMatch(/filled/);
        expect(stars[3].className).not.toMatch(/filled/);
        // aria-checked tracks your exact score.
        expect(stars[2].getAttribute('aria-checked')).toBe('true');
        expect(stars[1].getAttribute('aria-checked')).toBe('false');
    });

    it('renders the community average and pluralized count', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 4, rating_avg: 3.456 }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByText('3.5')).toBeInTheDocument();
        expect(screen.getByText(/\(4 ratings\)/)).toBeInTheDocument();
    });

    it('renders singular "rating" when count is 1', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 1, rating_avg: 5 }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByText(/\(1 rating\)/)).toBeInTheDocument();
    });

    it('shows the em-dash placeholder when avg is null but count > 0', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 2, rating_avg: null }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByText('—')).toBeInTheDocument();
    });

    it('previews on hover and clears on mouse leave', async () => {
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        fireEvent.mouseEnter(stars[3]); // hover 4th star => fill 1..4
        expect(stars[0].className).toMatch(/filled/);
        expect(stars[3].className).toMatch(/filled/);
        expect(stars[4].className).not.toMatch(/filled/);
        // Leave the stars container => hover resets.
        fireEvent.mouseLeave(stars[0].parentElement);
        stars.forEach((s) => expect(s.className).not.toMatch(/filled/));
    });

    it('singular aria-label for star 1, plural for others', async () => {
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        expect(screen.getByLabelText('1 star')).toBeInTheDocument();
        expect(screen.getByLabelText('2 stars')).toBeInTheDocument();
    });
});

describe('StarRating — rating actions', () => {
    it('calls rateAsset with the clicked score, updates UI, and fires onChange', async () => {
        const updated = summaryFor({ your_score: 4, rating_count: 1, rating_avg: 4 });
        rateAsset.mockResolvedValue({ data: updated });
        const onChange = vi.fn();
        await renderResolved({
            asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob', onChange,
        });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[3]); }); // 4th star
        expect(rateAsset).toHaveBeenCalledWith('rule', 'pub-1', 4);
        expect(onChange).toHaveBeenCalledWith(updated);
        await waitFor(() => expect(screen.getByText('Your rating')).toBeInTheDocument());
    });

    it('dispatches a gavel:ratingChanged event after a successful rate', async () => {
        rateAsset.mockResolvedValue({ data: summaryFor({ your_score: 2, rating_count: 1, rating_avg: 2 }) });
        const handler = vi.fn();
        window.addEventListener('gavel:ratingChanged', handler);
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[1]); });
        expect(handler).toHaveBeenCalledTimes(1);
        window.removeEventListener('gavel:ratingChanged', handler);
    });

    it('withdraws when clicking the same star you already rated', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ your_score: 3, rating_count: 1, rating_avg: 3 }),
        });
        withdrawRating.mockResolvedValue({ data: summaryFor({ your_score: null, rating_count: 0, rating_avg: null }) });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[2]); }); // same as your_score 3
        expect(withdrawRating).toHaveBeenCalledWith('rule', 'pub-1');
        expect(rateAsset).not.toHaveBeenCalled();
        await waitFor(() => expect(screen.getByText('Rate this')).toBeInTheDocument());
    });

    it('stops click propagation so parent card handlers do not fire', async () => {
        rateAsset.mockResolvedValue({ data: summaryFor({ your_score: 1, rating_count: 1, rating_avg: 1 }) });
        const parentClick = vi.fn();
        let utils;
        await act(async () => {
            utils = render(
                <div onClick={parentClick}>
                    <StarRating asset_type="rule" asset_public_id="pub-1" author_username="bob" />
                </div>,
            );
        });
        const stars = utils.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); });
        expect(parentClick).not.toHaveBeenCalled();
    });

    it('shows the API error detail when rating fails', async () => {
        rateAsset.mockRejectedValue({ response: { data: { detail: 'Rate limited' } } });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); });
        await waitFor(() => expect(screen.getByText('Rate limited')).toBeInTheDocument());
    });

    it('falls back to err.message when there is no response detail', async () => {
        rateAsset.mockRejectedValue(new Error('Network down'));
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); });
        await waitFor(() => expect(screen.getByText('Network down')).toBeInTheDocument());
    });

    it('falls back to the generic message when error has no detail or message', async () => {
        rateAsset.mockRejectedValue({});
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); });
        await waitFor(() => expect(screen.getByText('Could not save rating.')).toBeInTheDocument());
    });

    it('ignores a second click while a rating request is still in flight', async () => {
        // rateAsset that we resolve manually to keep loading=true.
        let resolveRate;
        rateAsset.mockReturnValue(new Promise((res) => { resolveRate = res; }));
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); }); // starts loading
        // Buttons are disabled while loading; fireEvent on a disabled button is a no-op,
        // but assert applyRating's loading guard regardless: only one call so far.
        fireEvent.click(stars[1]);
        expect(rateAsset).toHaveBeenCalledTimes(1);
        await act(async () => {
            resolveRate({ data: summaryFor({ your_score: 1, rating_count: 1, rating_avg: 1 }) });
        });
    });
});

describe('StarRating — author read-only mode', () => {
    it('renders disabled read-only stars filled to the rounded community avg', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 6, rating_avg: 3.6, your_score: null }),
        });
        // author_username matches current user (case-insensitive lowercased).
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'Alice' });
        expect(screen.getByText('Community rating')).toBeInTheDocument();
        const stars = screen.getAllByRole('radio');
        stars.forEach((s) => expect(s).toBeDisabled());
        // round(3.6) = 4 => first 4 filled.
        expect(stars[3].className).toMatch(/filled/);
        expect(stars[4].className).not.toMatch(/filled/);
        expect(stars[0].className).toMatch(/readonly/);
    });

    it('does not call rate endpoints when an author clicks a star', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 2, rating_avg: 4 }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'alice' });
        const stars = screen.getAllByRole('radio');
        await act(async () => { fireEvent.click(stars[0]); });
        expect(rateAsset).not.toHaveBeenCalled();
        expect(withdrawRating).not.toHaveBeenCalled();
    });

    it('does not preview on hover for authors', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 2, rating_avg: 1 }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'alice' });
        const stars = screen.getAllByRole('radio');
        fireEvent.mouseEnter(stars[4]); // would fill all 5 if hover applied
        // round(1) = 1 => only first filled despite hovering the 5th.
        expect(stars[1].className).not.toMatch(/filled/);
    });

    it('shows "No ratings yet" for an author with zero ratings', async () => {
        getRatingSummary.mockResolvedValue({
            data: summaryFor({ rating_count: 0, rating_avg: null }),
        });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'alice' });
        expect(screen.getByText('No ratings yet')).toBeInTheDocument();
    });

    it('is not author mode when no user is in localStorage', async () => {
        localStorage.clear(); // currentUser becomes null
        getRatingSummary.mockResolvedValue({ data: summaryFor() });
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        // Interactive label present => not author-gated.
        expect(screen.getByText('Rate this')).toBeInTheDocument();
        const stars = screen.getAllByRole('radio');
        expect(stars[0]).not.toBeDisabled();
    });
});

describe('StarRating — compact mode and error fetch fallback', () => {
    it('omits the label and adds the compact class in compact mode', async () => {
        const { container } = await renderResolved({
            asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob', compact: true,
        });
        expect(screen.queryByText('Rate this')).not.toBeInTheDocument();
        expect(container.querySelector('.star-rating.compact')).toBeTruthy();
    });

    it('compact skeleton also carries the compact class', () => {
        getRatingSummary.mockReturnValue(new Promise(() => {}));
        const { container } = render(
            <StarRating asset_type="rule" asset_public_id="pub-1" author_username="bob" compact />,
        );
        expect(container.querySelector('.star-rating.compact')).toBeTruthy();
        expect(screen.getByLabelText(/Loading ratings/i)).toBeInTheDocument();
    });

    it('falls back to a zeroed summary when the initial fetch rejects', async () => {
        getRatingSummary.mockRejectedValue(new Error('boom'));
        await renderResolved({ asset_type: 'rule', asset_public_id: 'pub-1', author_username: 'bob' });
        // Fallback summary => non-author, count 0 => "Be the first to rate".
        await waitFor(() => expect(screen.getByText('Be the first to rate')).toBeInTheDocument());
    });
});
