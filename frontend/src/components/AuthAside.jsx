// Shared brand / feature panel that sits on the left of the login + register
// pages on wide viewports. Hides below ~960px (see auth.css). Copy is grounded
// in actual product surfaces (rules, cognitive elements, calibration,
// realtime monitoring) — no marketing fluff that doesn't ship.

import { FiShield, FiLayers, FiTarget, FiActivity } from 'react-icons/fi';

const FEATURES = [
    {
        icon: <FiShield size={18} />,
        tone: 'tone-indigo',
        title: 'Multi-rule rule sets',
        desc: 'Compose rules and cognitive elements into auditable safety stacks.',
    },
    {
        icon: <FiLayers size={18} />,
        tone: 'tone-blue',
        title: 'Curated public library',
        desc: 'Bookmark and remix vetted rules and CEs from the community.',
    },
    {
        icon: <FiTarget size={18} />,
        tone: 'tone-violet',
        title: 'Calibrated thresholds',
        desc: 'Tune precision and recall against real evaluation runs, not guesses.',
    },
    {
        icon: <FiActivity size={18} />,
        tone: 'tone-emerald',
        title: 'Realtime monitoring',
        desc: 'Stream production traffic through the same rule set you trained.',
    },
];

const AuthAside = () => (
    <aside className="auth-aside">
        <div className="aside-brand">
            <div className="aside-mark"><FiShield size={24} /></div>
            <div>
                <div className="aside-brand-name">Gavel</div>
                <span className="aside-brand-kicker">Cloud Platform</span>
            </div>
        </div>

        <h2 className="aside-headline">
            Ship safer AI,<br />
            <span className="grad">faster.</span>
        </h2>
        <p className="aside-sub">
            Build, evaluate, and monitor production-grade safety rule sets from
            one place — with the rules, datasets, and calibration loops your team
            actually needs.
        </p>

        <ul className="aside-features">
            {FEATURES.map((f) => (
                <li key={f.title} className="aside-feature">
                    <span className={`aside-feature-icon ${f.tone}`}>{f.icon}</span>
                    <div className="aside-feature-text">
                        <div className="aside-feature-title">{f.title}</div>
                        <div className="aside-feature-desc">{f.desc}</div>
                    </div>
                </li>
            ))}
        </ul>

        <div className="aside-foot">
            <span className="aside-foot-dot" />
            Built for production AI safety teams
        </div>
    </aside>
);

export default AuthAside;
