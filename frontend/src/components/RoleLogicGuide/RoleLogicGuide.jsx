// RoleLogicGuide — the single source of truth for "how roles shape the firing
// boolean logic": the AND / OR / OPT legend plus worked examples. Shared by the
// Build-Rule wizard's "Learn Roles" step and the RuleCard "How it works" modal,
// so the explanation is identical everywhere a rule is shown or built.

import RuleLogicPreview from '../RuleLogicPreview/RuleLogicPreview';

const roleTag = (a, b) => ({
    background: `linear-gradient(135deg, ${a} 0%, ${b} 100%)`, color: '#fff', fontSize: '0.7rem',
    fontWeight: 700, padding: '2px 8px', borderRadius: 6, flexShrink: 0, marginTop: 2,
});

// Worked examples — real cognitive elements from the public library composed
// into believable rules, one per role pattern (Necessary / Any-of-group /
// Supporting). The title names the rule + the shape the chips below render.
const EXAMPLES = [
    {
        title: 'Credential phishing · Necessary AND Necessary',
        ces: [
            { name: 'click_or_enter', role: 'necessary' },
            { name: 'personal_information', role: 'necessary' },
        ],
    },
    {
        title: 'Targeted hate speech · Necessary AND Any-of group',
        ces: [
            { name: 'hatespeech', role: 'necessary' },
            { name: 'ethnoracial', role: 'fallback', fallback_group: 1 },
            { name: 'LGBTQ', role: 'fallback', fallback_group: 1 },
        ],
    },
    {
        title: 'Payment scam · Any-of group AND Any-of group',
        ces: [
            { name: 'send_or_transfer', role: 'fallback', fallback_group: 1 },
            { name: 'buy_or_purchase', role: 'fallback', fallback_group: 1 },
            { name: 'payment_tools', role: 'fallback', fallback_group: 2 },
            { name: 'personal_information', role: 'fallback', fallback_group: 2 },
        ],
    },
    {
        title: 'Financial scam · Necessary + Supporting',
        ces: [
            { name: 'send_or_transfer', role: 'necessary' },
            { name: 'trust_seeding', role: 'sufficient' },
        ],
    },
];

export default function RoleLogicGuide() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.14), rgba(99, 102, 241, 0.14))', border: '1px solid rgba(148, 163, 184, 0.18)', borderRadius: 14, padding: 18, lineHeight: 1.6 }}>
                <p style={{ margin: '0 0 10px', fontWeight: 700, color: '#f1f5f9' }}>How roles shape the predicate</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={roleTag('#a78bfa', '#8b5cf6')}>AND</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}><b>Necessary</b> — all must be present.</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={roleTag('#818cf8', '#3b82f6')}>OR</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}><b>Any of</b> — grouped alternatives. At least one per group.</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={roleTag('#34d399', '#10b981')}>OPT</span>
                        <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}><b>Supporting</b> — raises confidence when present, but does not trigger the rule on its own (not part of the boolean logic).</span>
                    </div>
                </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <p style={{ margin: 0, fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Examples</p>
                {EXAMPLES.map((ex) => (
                    <RuleLogicPreview key={ex.title} title={ex.title} ces={ex.ces} />
                ))}
            </div>
        </div>
    );
}
