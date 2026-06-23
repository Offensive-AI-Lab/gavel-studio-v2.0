// Tests for RuleLogicPreview — renders the firing boolean expression from CE
// role assignments (chips + AND/OR), with Supporting shown separately.
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RuleLogicPreview from '../../../src/components/RuleLogicPreview/RuleLogicPreview';

describe('RuleLogicPreview', () => {
    it('renders necessary CE chips joined by AND', () => {
        render(<RuleLogicPreview ces={[
            { name: 'alpha', role: 'necessary' },
            { name: 'beta', role: 'necessary' },
        ]} />);
        expect(screen.getByText('alpha')).toBeInTheDocument();
        expect(screen.getByText('beta')).toBeInTheDocument();
        expect(screen.getAllByText('AND').length).toBeGreaterThanOrEqual(1);
    });

    it('renders OR within an "any of" group', () => {
        render(<RuleLogicPreview ces={[
            { name: 'x', role: 'fallback', fallback_group: 1 },
            { name: 'y', role: 'fallback', fallback_group: 1 },
        ]} />);
        expect(screen.getByText('OR')).toBeInTheDocument();
    });

    it('lists Supporting CEs separately from the firing logic', () => {
        render(<RuleLogicPreview ces={[
            { name: 'a', role: 'necessary' },
            { name: 'sup', role: 'sufficient' },
        ]} />);
        expect(screen.getByText('sup')).toBeInTheDocument();
        expect(screen.getByText(/Supporting/)).toBeInTheDocument();
    });

    it('shows the empty hint when there is no firing logic', () => {
        render(<RuleLogicPreview ces={[{ name: 'sup', role: 'sufficient' }]} />);
        expect(screen.getByText(/Mark at least one CE/)).toBeInTheDocument();
    });
});
