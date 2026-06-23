// StarRating — Phase 3 widget for rating a published rule or CE.
//
// Three modes:
//   1. Not authenticated / no public_id            → not rendered.
//   2. Authenticated, NOT the artifact's author    → interactive 1-5 stars.
//      Hover preview, click to rate (or update), click the same star
//      again to withdraw.
//   3. Authenticated, IS the artifact's author     → read-only display.
//      Self-rating is blocked at the API; we render a muted star line
//      with the aggregate so the author can see how their work is
//      perceived without an "interactive but always rejected" UX.
//
// Props:
//   asset_type        : 'rule' | 'ce'
//   asset_public_id   : string (published artifacts only — drafts have no public_id yet)
//   author_username   : owner of the artifact; widget is read-only if this matches the current user
//   onChange          : optional callback fired after a successful rate / withdraw
//                       so the parent card can refresh aggregates if needed
//   compact           : if true, render in a single horizontal strip without label
//
// The widget fetches its own summary on mount, so a card containing
// many ratings widgets will do one GET per visible card. Future
// optimization: have Browse / Profile pre-fetch in batch and pass
// `initialSummary` down. Not needed for Phase 3 traffic levels.

import { useEffect, useState, useCallback } from 'react';
import { FiStar } from 'react-icons/fi';
import { getRatingSummary, rateAsset, withdrawRating } from '../../api';
import './StarRating.css';

const StarRating = ({
    asset_type,
    asset_public_id,
    author_username,
    onChange,
    compact = false,
}) => {
    const currentUser = JSON.parse(sessionStorage.getItem('user') || 'null');
    const isAuthor = !!currentUser
        && !!author_username
        && currentUser.username === author_username.toLowerCase();

    const [summary, setSummary] = useState(null);
    const [loading, setLoading] = useState(false);
    const [hover, setHover] = useState(0);
    const [error, setError] = useState(null);

    // Initial fetch. asset_public_id is the trigger — drafts have no
    // public_id and the widget shouldn't render in that case, but we
    // double-check here too so a misuse just no-ops instead of 404ing.
    useEffect(() => {
        if (!asset_public_id) return;
        let cancelled = false;
        getRatingSummary(asset_type, asset_public_id).then((res) => {
            if (!cancelled) setSummary(res.data);
        }).catch(() => {
            if (!cancelled) setSummary({
                asset_type, asset_public_id,
                rating_count: 0, rating_avg: null, your_score: null,
            });
        });
        return () => { cancelled = true; };
    }, [asset_type, asset_public_id]);

    const applyRating = useCallback(async (score) => {
        if (!asset_public_id || loading) return;
        setLoading(true);
        setError(null);
        try {
            // If they click the same star they already rated, treat
            // that as withdrawal. Same gesture as Spotify's "remove
            // from liked". Otherwise upsert with the new score.
            let res;
            if (summary?.your_score === score) {
                res = await withdrawRating(asset_type, asset_public_id);
            } else {
                res = await rateAsset(asset_type, asset_public_id, score);
            }
            setSummary(res.data);
            if (onChange) onChange(res.data);
            // Notify any listening page (Profile, Browse) that a rating
            // changed so they can re-fetch aggregate data (e.g., the
            // profile header's "avg rating received"). Lighter than
            // withNotify (which triggers a full library refresh and
            // collapses cards); this only signals "rating data changed."
            window.dispatchEvent(new Event('gavel:ratingChanged'));
        } catch (err) {
            const msg = err.response?.data?.detail || err.message || 'Could not save rating.';
            setError(msg);
        } finally {
            setLoading(false);
        }
    }, [asset_type, asset_public_id, summary, loading, onChange]);

    if (!asset_public_id) return null;
    if (!summary) {
        return (
            <div className={`star-rating ${compact ? 'compact' : ''}`}>
                <span className="star-rating-skeleton" aria-label="Loading ratings…" />
            </div>
        );
    }

    const yourScore = summary.your_score || 0;
    const avg = summary.rating_avg;
    const count = summary.rating_count;
    // Star fill semantics:
    //   * Authors can't rate their own work → stars show the community avg
    //     (rounded), and the widget is read-only.
    //   * Non-authors: stars represent YOUR rating, not the community's.
    //     Empty stars when you haven't rated; your score when you have;
    //     hover preview while you're picking a new score.
    //   The numeric "X.X (N ratings)" string next to the stars still
    //   shows the community average — see the rating-summary text below.
    const displayedFill = isAuthor
        ? Math.round(avg || 0)
        : (hover || yourScore);

    return (
        <div className={`star-rating ${compact ? 'compact' : ''}`} aria-live="polite">
            {!compact && (
                <span className="star-rating-label">
                    {isAuthor ? 'Community rating' : (yourScore ? 'Your rating' : 'Rate this')}
                </span>
            )}
            <div
                className="star-rating-stars"
                onMouseLeave={() => setHover(0)}
                role="radiogroup"
                aria-label={`Rate this ${asset_type}, 1 to 5 stars`}
            >
                {[1, 2, 3, 4, 5].map((n) => {
                    const filled = n <= displayedFill;
                    return (
                        <button
                            key={n}
                            type="button"
                            className={`star-rating-star ${filled ? 'filled' : ''} ${isAuthor ? 'readonly' : ''}`}
                            onMouseEnter={() => !isAuthor && setHover(n)}
                            onClick={(e) => {
                                e.stopPropagation();
                                if (!isAuthor) applyRating(n);
                            }}
                            disabled={isAuthor || loading}
                            aria-label={`${n} star${n === 1 ? '' : 's'}`}
                            aria-checked={yourScore === n}
                            role="radio"
                            title={
                                isAuthor
                                    ? `${avg ? avg.toFixed(1) : '—'} avg from ${count} rating${count === 1 ? '' : 's'}`
                                    : (yourScore === n ? 'Click again to remove your rating' : `Rate ${n}`)
                            }
                        >
                            <FiStar />
                        </button>
                    );
                })}
            </div>
            <div className="star-rating-meta">
                {count > 0 ? (
                    <>
                        <strong>{avg ? avg.toFixed(1) : '—'}</strong>
                        <span className="star-rating-count">
                            ({count} rating{count === 1 ? '' : 's'})
                        </span>
                    </>
                ) : (
                    <span className="star-rating-count">
                        {isAuthor ? 'No ratings yet' : 'Be the first to rate'}
                    </span>
                )}
            </div>
            {error && <span className="star-rating-error">{error}</span>}
        </div>
    );
};

export default StarRating;
