// Tests for the rule predicate derivation — must mirror the backend's
// _build_predicate_from_roles exactly (the live preview shows what publishes).
import { describe, it, expect } from 'vitest';
import { partitionRoles, buildPredicateString } from '../../src/utils/rulePredicate';

describe('buildPredicateString', () => {
    it('AND-joins necessary CEs', () => {
        expect(buildPredicateString([
            { name: 'a', role: 'necessary' },
            { name: 'b', role: 'necessary' },
        ])).toBe('a AND b');
    });

    it('wraps an "any of" group and ANDs it with the necessary part', () => {
        expect(buildPredicateString([
            { name: 't', role: 'necessary' },
            { name: 'x', role: 'fallback', fallback_group: 1 },
            { name: 'y', role: 'fallback', fallback_group: 1 },
        ])).toBe('t AND (x OR y)');
    });

    it('ANDs multiple groups together in group-id order', () => {
        expect(buildPredicateString([
            { name: 'x', role: 'fallback', fallback_group: 2 },
            { name: 'y', role: 'fallback', fallback_group: 1 },
        ])).toBe('(y) AND (x)');
    });

    it('excludes Supporting CEs from the firing predicate', () => {
        expect(buildPredicateString([
            { name: 'a', role: 'necessary' },
            { name: 's', role: 'sufficient' },
        ])).toBe('a');
    });

    it('partitions roles into necessary / groups / supporting', () => {
        const { necessary, groups, supporting } = partitionRoles([
            { name: 'a', role: 'necessary' },
            { name: 'x', role: 'fallback', fallback_group: 1 },
            { name: 'y', role: 'fallback', fallback_group: 1 },
            { name: 's', role: 'sufficient' },
        ]);
        expect(necessary).toEqual(['a']);
        expect(groups).toEqual([['x', 'y']]);
        expect(supporting).toEqual(['s']);
    });
});
