// Display-only labels for CE roles, shown in rule cards, the rule wizard, and
// role legends. The INTERNAL role values (necessary / fallback / sufficient)
// and the backend / DB are unchanged — this maps each internal value to the
// name the user sees. Change a label here and it updates everywhere.
//   necessary  -> "Necessary"
//   fallback   -> "Any of"
//   sufficient -> "Supporting"
export const ROLE_LABELS = {
    necessary: 'Necessary',
    fallback: 'Any of',
    sufficient: 'Supporting',
};

// Display label for a role value (defaults to Necessary for unknown/missing).
export const roleLabel = (role) => ROLE_LABELS[role] || ROLE_LABELS.necessary;

// "Any of · G2" — `group0` is the 0-indexed DB fallback_group, shown 1-indexed.
export const anyOfGroupLabel = (group0) => `${ROLE_LABELS.fallback} · G${(group0 || 0) + 1}`;
