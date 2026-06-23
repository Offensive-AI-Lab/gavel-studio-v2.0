// Derive a rule's firing predicate from CE role assignments — mirrors the
// backend's _build_predicate_from_roles EXACTLY so the live preview matches what
// actually gets published:
//   * Necessary CEs are AND-joined.
//   * Each "Any of" (fallback) group becomes "(a OR b)", and the groups are
//     AND-joined together.
//   * Supporting CEs are NEVER part of the firing predicate (helpful signal only).
//
// Input shape: an array of { name, role, fallback_group }.

export function partitionRoles(ces) {
    const necessary = [];
    const groups = {};        // group id -> [names]
    const supporting = [];
    (ces || []).forEach(({ name, role, fallback_group }) => {
        const r = role || 'necessary';
        if (r === 'fallback') {
            const g = parseInt(fallback_group, 10) || 0;
            (groups[g] = groups[g] || []).push(name);
        } else if (r === 'sufficient') {
            supporting.push(name);
        } else {
            necessary.push(name);
        }
    });
    // Sort groups by id so the order is stable + matches the backend.
    const groupList = Object.keys(groups)
        .map(Number)
        .sort((a, b) => a - b)
        .map((g) => groups[g]);
    return { necessary, groups: groupList, supporting };
}

// The plain-text predicate (e.g. "a AND (b OR c)"), identical to the backend's.
export function buildPredicateString(ces) {
    const { necessary, groups } = partitionRoles(ces);
    const parts = [];
    if (necessary.length) parts.push(necessary.join(' AND '));
    groups.forEach((g) => { if (g.length) parts.push('(' + g.join(' OR ') + ')'); });
    return parts.join(' AND ');
}
