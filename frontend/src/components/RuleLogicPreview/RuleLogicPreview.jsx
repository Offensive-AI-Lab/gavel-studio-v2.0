// RuleLogicPreview — renders a rule's firing boolean expression as styled chips
// + AND/OR operators, derived live from CE role assignments. Used in the Build-
// Rule wizard: as a real-time preview while assigning roles, as worked examples
// on the "Learn Roles" step, and in the final summary.
//
// The logic mirrors the backend exactly (see utils/rulePredicate): Necessary
// AND-joined, each "Any of" group "(a OR b)" AND-joined, Supporting excluded
// from the firing logic (shown separately, muted).

import { partitionRoles } from '../../utils/rulePredicate';

const COLORS = {
    necessary: '#a78bfa',   // violet  (AND)
    fallback:  '#818cf8',   // indigo  (OR group)
    op:        '#8b5cf6',
};

const chipStyle = (color) => ({
    display: 'inline-flex', alignItems: 'center',
    padding: '4px 11px', borderRadius: 999,
    fontSize: '0.82rem', fontWeight: 700, color: '#f8fafc',
    background: `${color}26`, border: `1px solid ${color}`,
    whiteSpace: 'nowrap',
});

const opStyle = {
    fontSize: '0.7rem', fontWeight: 800, letterSpacing: '0.06em',
    color: '#cbd5e1', padding: '0 2px',
};

const parenStyle = {
    fontSize: '1.1rem', fontWeight: 800, color: '#64748b', alignSelf: 'center',
};

export default function RuleLogicPreview({
    ces,
    title = 'Boolean logic — live preview',
    emptyHint = 'Mark at least one CE as Necessary or Any of to form the firing logic.',
    style = {},
}) {
    const { necessary, groups, supporting } = partitionRoles(ces);
    const hasLogic = necessary.length > 0 || groups.some((g) => g.length > 0);

    const nodes = [];
    let first = true;
    const and = (k) => nodes.push(<span key={k} style={opStyle}>AND</span>);

    necessary.forEach((n, i) => {
        if (!first) and(`and-n-${i}`);
        nodes.push(<span key={`n-${i}`} style={chipStyle(COLORS.necessary)}>{n}</span>);
        first = false;
    });
    groups.forEach((g, gi) => {
        if (!g.length) return;
        if (!first) and(`and-g-${gi}`);
        nodes.push(<span key={`po-${gi}`} style={parenStyle}>(</span>);
        g.forEach((n, i) => {
            if (i > 0) nodes.push(<span key={`or-${gi}-${i}`} style={opStyle}>OR</span>);
            nodes.push(<span key={`g-${gi}-${i}`} style={chipStyle(COLORS.fallback)}>{n}</span>);
        });
        nodes.push(<span key={`pc-${gi}`} style={parenStyle}>)</span>);
        first = false;
    });

    return (
        <div style={{
            background: 'rgba(2, 6, 23, 0.55)',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            borderRadius: 12, padding: '12px 14px', ...style,
        }}>
            {title && (
                <div style={{
                    fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase',
                    letterSpacing: '0.05em', color: '#94a3b8', marginBottom: 10,
                }}>{title}</div>
            )}
            {hasLogic ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
                    {nodes}
                </div>
            ) : (
                <div style={{ fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic' }}>{emptyHint}</div>
            )}
            {supporting.length > 0 && (
                <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#34d399' }}>+ Supporting</span>
                    <span style={{ fontSize: '0.74rem', color: '#64748b' }}>(raises confidence, not part of the logic):</span>
                    {supporting.map((n, i) => (
                        <span key={`s-${i}`} style={{ ...chipStyle('#34d399'), opacity: 0.85 }}>{n}</span>
                    ))}
                </div>
            )}
        </div>
    );
}
